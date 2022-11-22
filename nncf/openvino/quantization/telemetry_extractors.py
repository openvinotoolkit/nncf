from typing import Any

from nncf.telemetry.extractors import CollectedEvent
from nncf.telemetry.extractors import TelemetryExtractor


class CompressionStarted(TelemetryExtractor):
    def extract(self, _: Any) -> CollectedEvent:
        return CollectedEvent(name="compression_started",
                              data="DefaultQuantization")

