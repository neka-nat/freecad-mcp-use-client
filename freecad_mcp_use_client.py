import argparse
import asyncio
import base64

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
from mcp_use import MCPAgent, MCPClient


async def main(prompt: str, image_path: str):
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

    llm = ChatOpenAI(model="o3")
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
    args = parser.parse_args()
    asyncio.run(main(args.prompt, args.image_path))
