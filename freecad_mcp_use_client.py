import argparse
import asyncio
import base64

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage
from mcp_use import MCPAgent, MCPClient


async def main(prompt: str, model: str, image_path: str):
    load_dotenv()

    config = {
      "mcpServers": {
        "freecad": {
          "command": "uvx",
          "args": ["freecad-mcp"]
        }
      }
    }

    client = MCPClient.from_dict(config)

    if model == "gpt":
        llm = ChatOpenAI(model="gpt-4.1-mini-2025-04-14")
    elif model == "genai":
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
    else:
        raise ValueError(f"Invalid model: {model}")
    agent = MCPAgent(llm=llm, client=client, max_steps=30)

    with open(image_path, "rb") as image_file:
        image = base64.b64encode(image_file.read()).decode("utf-8")

    result = await agent.run(
        prompt,
        external_history=[
            HumanMessage(content=[
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/{image_path.split('.')[-1]};base64,{image}"
                    }
                }
            ])
        ] if image_path else []
    )
    print(f"\nResult: {result}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="Create a 3D model of the attached image.")
    parser.add_argument("--image_path", type=str)
    parser.add_argument("--model", type=str, default="gpt")
    args = parser.parse_args()
    asyncio.run(main(args.prompt, args.model, args.image_path))
