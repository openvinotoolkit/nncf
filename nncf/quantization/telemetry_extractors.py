from typing import Any

from nncf.telemetry.extractors import CollectedEvent
from nncf.telemetry.extractors import TelemetryExtractor


class CompressionStartedWithQuantizeApi(TelemetryExtractor):
    def extract(self, _: Any) -> CollectedEvent:
        return CollectedEvent(name="compression_started",
                              data="quantize_api")
