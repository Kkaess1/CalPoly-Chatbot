import json
import boto3
import os
import logging
import re
from botocore.config import Config

# --- Initialization & Logging ---
logger = logging.getLogger()
logger.setLevel(logging.INFO)

config = Config(
    retries={'max_attempts': 3, 'mode': 'adaptive'},
    connect_timeout=5,
    read_timeout=60
)

bedrock_agent = boto3.client(
    'bedrock-agent-runtime',
    region_name=os.environ['REGION'],
    config=config
)

# Environment variables — MODEL_ID must be the full ARN
MODEL_ID          = os.environ['MODEL_ID']
KNOWLEDGE_BASE_ID = os.environ['KNOWLEDGE_BASE_ID']
ALLOWED_ORIGIN    = os.environ.get('ALLOWED_ORIGIN', '*')


# --- Helpers ---

def clean_urls(text):
    """Strip trailing punctuation from URLs so they stay clickable."""
    return re.sub(
        r'(https?://[^\s]+)',
        lambda m: m.group(0).rstrip('.,!?)]}'),
        text
    )


def build_prompt(user_message):
    return """
You are PolyBot, a friendly and knowledgeable assistant for Cal Poly's Master of Science in Business Analytics (MSBA) program.
Answer student questions using only the retrieved knowledge base content provided to you.

<instructions>
1. Answer using only retrieved content — do not invent, assume, or infer information.
2. Keep responses conversational, clear, and concise (100 words or fewer, expanding only when genuinely necessary).
3. Only include URLs that appear exactly in the retrieved content — never fabricate or modify them.
4. Do not add punctuation immediately after any URL.
5. If a question is unclear, off-topic, or not covered by retrieved content, politely say so and point the user to a relevant resource below.
6. You have full context of the current conversation — use prior questions and answers to give contextually aware follow-up responses.

Useful resources (only cite if relevant):
- Contact Form: https://forms.office.com/pages/responsepage.aspx?id=2wING578lUSVNx03nMoq5_unPySOHhVKs6D8naanJd5UMkswTFVONVBRME02WllTWjdSSjJCU0o2OS4u&route=shorturl
- FAQs: https://orfalea.calpoly.edu/graduate-programs/ms-business-analytics/faqs
- Information Sessions: https://orfalea.calpoly.edu/graduate-programs/ms-business-analytics#information-sessions

For prerequisite questions: students must have completed one college-level calculus course and two college-level statistics courses.
</instructions>

<examples>
Q: What are the prerequisites for the MSBA program?
A: To apply, you'll need one college-level calculus course and two college-level statistics courses. For more details, check the FAQs: https://orfalea.calpoly.edu/graduate-programs/ms-business-analytics/faqs

Q: Can you tell me about parking on campus?
A: That falls outside what I can help with here! For MSBA-unrelated questions, reach out directly: https://forms.office.com/pages/responsepage.aspx?id=2wING578lUSVNx03nMoq5_unPySOHhVKs6D8naanJd5UMkswTFVONVBRME02WllTWjdSSjJCU0o2OS4u&route=shorturl
</examples>

Question: """ + user_message + """
Answer:""".strip()


# --- Main Lambda Handler ---

def handler(event, context):
    logger.info("Received request")

    method = (
        event.get('httpMethod') or
        event.get('requestContext', {}).get('http', {}).get('method') or
        event.get('requestContext', {}).get('httpMethod')
    )

    headers = {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Headers': 'Content-Type',
        'Access-Control-Allow-Methods': 'POST, OPTIONS',
        'Content-Type': 'application/json'
    }

    # Handle CORS preflight
    if method == 'OPTIONS':
        return {'statusCode': 200, 'headers': headers, 'body': ''}

    try:
        # 1. Parse request
        body         = json.loads(event.get('body', '{}'))
        user_message = body.get('message', '').strip()

        # Only use sessionId if it was returned by Bedrock in a prior response.
        # Never pass a client-generated UUID — Bedrock will reject it.
        bedrock_session_id = body.get('bedrockSessionId', '').strip()

        logger.info("User message: %s", user_message)
        logger.info("Bedrock session ID received: %s", bedrock_session_id or "(none — first message)")

        # Empty input guard
        if not user_message:
            return {
                'statusCode': 200,
                'headers': headers,
                'body': json.dumps({
                    'answer': 'Hi! How can I help you with the Cal Poly MSBA program today?'
                })
            }

        # 2. Build RAG request — only add sessionId if Bedrock gave us one previously
        rag_params = {
            'input': {'text': build_prompt(user_message)},
            'retrieveAndGenerateConfiguration': {
                'type': 'KNOWLEDGE_BASE',
                'knowledgeBaseConfiguration': {
                    'knowledgeBaseId': KNOWLEDGE_BASE_ID,
                    'modelArn': MODEL_ID
                }
            }
        }

        if bedrock_session_id:
            rag_params['sessionId'] = bedrock_session_id

        # 3. Call Bedrock
        response = bedrock_agent.retrieve_and_generate(**rag_params)

        # 4. Extract answer and Bedrock's session ID
        raw_answer         = response['output']['text']
        returned_session_id = response.get('sessionId', '')
        final_answer        = clean_urls(raw_answer)

        logger.info("Bedrock session ID returned: %s", returned_session_id)
        logger.info("Answer generated successfully")

        return {
            'statusCode': 200,
            'headers': headers,
            'body': json.dumps({
                'answer':           final_answer,
                'bedrockSessionId': returned_session_id  # frontend stores and sends this back
            })
        }

    except Exception as e:
        logger.error("Error processing request: %s", str(e), exc_info=True)
        return {
            'statusCode': 500,
            'headers': headers,
            'body': json.dumps({
                'error': 'Unable to process your request. Please try again.'
            })
        }
