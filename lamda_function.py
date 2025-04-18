import json
import os
import time
import boto3
import requests

dynamodb = boto3.client("dynamodb")
TABLE_NAME = "chat-history"

# CORS headers
cors_headers = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type, Authorization"
}

def lambda_handler(event, context):
    # Handle OPTIONS preflight request
    if event.get("requestContext", {}).get("http", {}).get("method") == "OPTIONS":
        return {
            "statusCode": 200,
            "headers": cors_headers,
            "body": json.dumps({"message": "CORS preflight success"})
        }

    if event.get("requestContext", {}).get("http", {}).get("method") != "POST":
        return {
            "statusCode": 405,
            "headers": cors_headers,
            "body": json.dumps({"error": "Method Not Allowed. Use POST."})
        }

    try:
        body = json.loads(event.get("body", "{}"))
        user_prompt = body["user_prompt"]
        user_id = body.get("user_id", "default_user")  # Optional: replace with real user ID
    except (json.JSONDecodeError, KeyError):
        return {
            "statusCode": 400,
            "headers": cors_headers,
            "body": json.dumps({"error": "Invalid request. Missing 'user_prompt'."})
        }

    try:
        # 1. Fetch last 6 messages from DynamoDB (for this user)
        history = []
        try:
            response = dynamodb.query(
                TableName=TABLE_NAME,
                KeyConditionExpression="user_id = :uid",
                ExpressionAttributeValues={":uid": {"S": user_id}},
                ScanIndexForward=False,
                Limit=6
            )
            items = reversed(response.get("Items", []))  # Oldest first
            history = [item["content"]["S"] for item in items]
        except Exception as e:
            print("DynamoDB read error:", e)

        # 2. Format prompt using history + new message
        prompt = "<s>"
        for i, msg in enumerate(history):
            if i % 2 == 0:  # user message
                prompt += f"[INST] {msg} [/INST]"
            else:  # bot message
                prompt += f" {msg} </s><s>"
        prompt += f"[INST] {user_prompt} [/INST]"

        # 3. Send to Hugging Face
        hf_token = os.getenv("HF_API_TOKEN")
        headers = {"Authorization": f"Bearer {hf_token}"}

        response = requests.post(
            "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3",
            headers=headers,
            json={"inputs": prompt}
        )

        result = response.json()
        if isinstance(result, dict) and "error" in result:
            raise Exception(result["error"])

        # Clean AI reply
        raw_reply = result[0]["generated_text"]
        ai_reply = raw_reply.split("[/INST]")[-1].strip()

        # 4. Save user message + bot reply to DynamoDB
        timestamp = int(time.time() * 1000)
        dynamodb.put_item(
            TableName=TABLE_NAME,
            Item={
                "user_id": {"S": user_id},
                "timestamp": {"N": str(timestamp)},
                "content": {"S": user_prompt}
            }
        )
        dynamodb.put_item(
            TableName=TABLE_NAME,
            Item={
                "user_id": {"S": user_id},
                "timestamp": {"N": str(timestamp + 1)},
                "content": {"S": ai_reply}
            }
        )

        return {
            "statusCode": 200,
            "headers": cors_headers,
            "body": json.dumps({"ai_reply": ai_reply})
        }

    except Exception as e:
        return {
            "statusCode": 500,
            "headers": cors_headers,
            "body": json.dumps({"error": f"Hugging Face API error: {str(e)}"})
        }
