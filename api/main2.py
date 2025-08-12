import httpx

async def test_stream():
    async with httpx.AsyncClient() as client:
        async with client.stream(
            "POST",
            "http://127.0.0.1:8000/prompt-stream",
            json={"prompt": "What did Nietzsche say about morality?", "n_results": 3}
        ) as response:
            async for chunk in response.aiter_text():
                print(chunk, end="")

# Run the test
import asyncio
asyncio.run(test_stream())