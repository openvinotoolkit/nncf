import cpuinfo
import re


def _is_lnl_cpu() -> bool:
    return re.search(r'Ultra \d 2\d{2}V', cpuinfo.get_cpu_info()["brand_raw"]) is not None


IS_LNL_CPU = _is_lnl_cpu()
