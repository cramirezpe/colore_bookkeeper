"""Script to run Corrf and CoLoRe"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from colore_bookkeeper.bookkeeper import Bookkeeper
from colore_bookkeeper.scripts.run_CoLoRe import main as run_CoLoRe

if TYPE_CHECKING:
    from typing import Optional
logger = logging.getLogger(__name__)

def main(args: Optional[argparse.Namespace] = None) -> None:
    if args is None:
        args = get_args()

    level = logging.getLevelName(args.log_level)
    logging.basicConfig(
        stream=sys.stdout,
        level=level,
        format="%(levelname)s:%(message)s",
    )

    colore_args = argparse.Namespace(
        bookkeeper_config=args.bookkeeper_config,
        overwrite=args.overwrite,
        skip_sent=args.skip_sent,
        only_write=args.only_write,
        wait_for=args.wait_for,
        log_level=args.log_level,
        overwrite_config=args.overwrite_config,
    )

    run_CoLoRe(colore_args)

    bookkeeper = Bookkeeper(
        args.bookkeeper_config,
        overwrite_config=args.overwrite_config,
        read_mode=False,
    )
    
    logger.info("Adding Corrf.")
    corrf = bookkeeper.get_corrf_tasker(
        wait_for=args.wait_for,
        overwrite=args.overwrite,
        skip_sent=args.skip_sent,
    )

    corrf.write_job()
    if not args.only_write:
        corrf.send_job()
        logger.info(f"Sent Corrf run:\n\t{corrf.jobid}")


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "bookkeeper_config", type=Path, help="Path to bookkeeper file to use"
    )

    parser.add_argument(
        "--overwrite", action="store_true", help="Force overwrite output data."
    )

    parser.add_argument(
        "--overwrite-config",
        action="store_true",
        help="Force overwrite bookkeeper config.",
    )

    parser.add_argument(
        "--skip-sent", action="store_true", help="Skip runs that were already sent."
    )

    parser.add_argument("--wait-for", nargs="+", type=int, default=None, required=False)

    parser.add_argument(
        "--only-write",
        action="store_true",
        help="Only write scripts, do not send them.",
    )

    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"],
    )

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    main()
