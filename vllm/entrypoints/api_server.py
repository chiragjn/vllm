import argparse
import json
from typing import AsyncGenerator

from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
import uvicorn

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.outputs import CompletionOutput
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid

TIMEOUT_KEEP_ALIVE = 5  # seconds.
TIMEOUT_TO_PREVENT_DEADLOCK = 1  # seconds.
app = FastAPI()


@app.post("/generate")
async def generate(request: Request) -> Response:
    """Generate completion for the request.

    The request should be a JSON object with the following fields:
    - prompt: the prompt to use for the generation.
    - stream: whether to stream the results or not.
    - other fields: the sampling parameters (See `SamplingParams` for details).
    """
    
    request_dict = await request.json()
    if "prompt" in request_dict:
        request_format = "vllm"
        prompt = request_dict.pop("prompt")
        stream = request_dict.pop("stream", False)
        sampling_params = SamplingParams(**request_dict)
    elif "inputs" in request_dict:
        request_format = "tgi"
        prompt = request_dict.pop("inputs")
        stream = request_dict.pop("stream", False)
        sampling_params = request_dict.pop("parameters") or {}
        if "max_new_tokens" in sampling_params:
            sampling_params["max_tokens"] = sampling_params.pop("max_new_tokens")
        if "n" in sampling_params and sampling_params["n"] > 1:
            return JSONResponse({"error": "n cannot be greater than 1"}, status_code=400)
        sampling_params = SamplingParams(**sampling_params)
    else:
        request_format = "unknown"

    request_id = random_uuid()
    results_generator = engine.generate(prompt, sampling_params, request_id)

    # Streaming case
    async def stream_results() -> AsyncGenerator[bytes, None]:
        async for request_output in results_generator:
            # TODO (chiragjn): Fix compatibility here!
            prompt = request_output.prompt
            text_outputs = [
                prompt + output.text for output in request_output.outputs
            ]
            ret = {"text": text_outputs}
            yield (json.dumps(ret) + "\0").encode("utf-8")

    async def abort_request() -> None:
        await engine.abort(request_id)

    if stream:
        background_tasks = BackgroundTasks()
        # Abort the request if the client disconnects.
        background_tasks.add_task(abort_request)
        return StreamingResponse(stream_results(), background=background_tasks)

    # Non-streaming case
    final_output = None
    async for request_output in results_generator:
        if await request.is_disconnected():
            # Abort the request if the client disconnects.
            await engine.abort(request_id)
            return Response(status_code=499)
        final_output = request_output

    assert final_output is not None
    prompt = final_output.prompt

    if request_format == "vllm":
        text_outputs = [prompt + output.text for output in final_output.outputs]
        ret = {"text": text_outputs}
    elif request_format == "tgi":
        ret = {}
        output: CompletionOutput = final_output.outputs[0]
        generated_text = output.text
        if sampling_params.return_full_text:
            generated_text = prompt + output.text
        ret["generated_text"] = generated_text

        details = {}
        if sampling_params.details:
            details["finish_reason"] = output.finish_reason
            details["generated_tokens"] = len(output.token_ids)
            # TODO (chiragjn): prefill and tokens is missing
            ret["details"] = details
    else:
        return JSONResponse({"error": "Unknown request format"}, status_code=400)
    return JSONResponse(ret)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser = AsyncEngineArgs.add_cli_args(parser)
    args = parser.parse_args()

    engine_args = AsyncEngineArgs.from_cli_args(args)
    engine = AsyncLLMEngine.from_engine_args(engine_args)

    uvicorn.run(app,
                host=args.host,
                port=args.port,
                log_level="debug",
                timeout_keep_alive=TIMEOUT_KEEP_ALIVE)
