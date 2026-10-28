from opentelemetry import metrics, trace

class OTelMiddleware(BaseMiddleware):
    """OpenTelemetry middleware following Gen-AI conventions."""

    def __init__(self):
        self._tracer = trace.get_tracer("littleagents")
        self._meter = metrics.get_meter("littleagents")

        # Create metric instruments following Gen-AI semconv
        self._token_histogram = self._meter.create_histogram(
            name="gen_ai.client.token.usage",
            unit="{token}",
            description="Token usage per request"
        )

        self._duration_histogram = self._meter.create_histogram(
            name="gen_ai.client.operation.duration",
            unit="s",
            description="Operation duration"
        )

    async def process_request(
        self,
        context: MiddlewareContext
    ) -> MiddlewareContext:
        """Start span before operation."""
        # Create span following Gen-AI naming conventions
        if context.operations == "model_call":
            span_name = f"chat {self._get_model_name(context)}"
        elif context.operation == "tool_call":
            span_name = f"tool {self._get_tool_name(context)}"

        span = self._tracer.start_span(span_name)

        # Add Gen-AI semantic convention attributes
        span.set_attribute("gen_ai.system", "littleagents")
        span.set_attribute("gen_ai.operation.name", context.operation)
        span.set_attribute("gen_ai.agent.name", context.agent_name)

        # Store span for completion in process_response
        context.metadata["_otel_span"] = span
        context.metadata["_otel_start"] = time.time()

        return context

    async def process_response(
        self,
        context: MiddlewareContext,
        result: Any
    ) -> Any:
        """Record metrics and end span."""
        span = context.metadata.get("_otel_span")

        # Record toke usage metrics for model calls
        if hasattr(result, "usage"):
            self._token_histogram.record(
                result.usage.prompt_tokens,
                {"gen_ai.token.type": "input"}
            )
            self._token_histogram.record(
                result.usage.completion_tokens,
                {"gen_ai.token.type": "output"}
            )

            # Add usage as span attributes
            span.set_attribute("gen_ai.usage.input_tokens",
                               result.usage.prompt_tokens)
            span.set_attribute("gen_ai.usage.output_tokens",
                               result.usage.completion_tokens)

        # Record duration and end span
        duration = time.time() - context.metadata["_otel_start"]
        self._duration_histogram.record(duration)
        span.set_status(Status(StatusCode.OK))
        span.end()
        return result
