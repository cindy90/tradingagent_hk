"""同花顺 iFinD 客户端封装（骨架）。

iFinD Python SDK 通常通过 `from iFinDPy import *` 引入，需先 THS_iFinDLogin。
此处仅勾出接口，待 SDK 接入后填充实际调用。未配置时所有方法返回空。
"""
from __future__ import annotations

from typing import Any

from loguru import logger

from config import get_settings

from .cache import disk_cache


class THSClient:
    def __init__(self) -> None:
        s = get_settings()
        self.user = s.ths_user
        self.password = s.ths_password
        self._logged_in = False
        self._sdk = None

    def _ensure_login(self) -> bool:
        if self._logged_in:
            return True
        if not (self.user and self.password):
            logger.debug("THS 未配置账号，跳过")
            return False
        try:
            from iFinDPy import THS_iFinDLogin  # type: ignore
            ret = THS_iFinDLogin(self.user, self.password)
            if ret == 0 or ret == -201:  # 0=成功, -201=已登录
                self._logged_in = True
                import iFinDPy  # type: ignore
                self._sdk = iFinDPy
                return True
            logger.warning(f"THS 登录失败 ret={ret}")
            return False
        except ImportError:
            logger.warning("iFinDPy SDK 未安装，THS 接口不可用")
            return False
        except Exception as e:
            logger.warning(f"THS 登录异常: {e}")
            return False

    @disk_cache(ttl_seconds=24 * 3600, namespace="ths")
    def get_basic_info(self, ticker: str) -> dict[str, Any]:
        """公司基础信息。ticker 港股形如 '00700.HK'。"""
        if not self._ensure_login():
            return {}
        # TODO: THS_BD(ticker, "ths_corp_chi_name_stock;ths_industry_gn_stock;...")
        return {}

    @disk_cache(ttl_seconds=7 * 24 * 3600, namespace="ths")
    def get_prospectus_pdf_url(self, ticker: str) -> str | None:
        """从 iFinD 获取招股书 PDF 下载链接。返回 URL 或 None。"""
        if not self._ensure_login():
            return None
        # TODO: THS_DR("p03393", ...)  招股书数据集编码视实际订阅而定
        return None

    @disk_cache(ttl_seconds=24 * 3600, namespace="ths")
    def get_industry_research(self, industry_code: str) -> list[dict]:
        """行业研究报告列表（标题/摘要/日期/作者）。"""
        if not self._ensure_login():
            return []
        # TODO: THS_RD 行业研报接口
        return []

    @disk_cache(ttl_seconds=24 * 3600, namespace="ths")
    def get_macro_indicators(self, indicators: list[str]) -> dict[str, list]:
        """宏观指标：CPI、PMI、HIBOR、港币兑美元等。"""
        if not self._ensure_login():
            return {}
        # TODO: THS_EDB 宏观数据库接口
        return {ind: [] for ind in indicators}
