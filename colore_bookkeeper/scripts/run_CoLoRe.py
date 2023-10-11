"""Script to run CoLoRe given a bookkeeper config file."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from colore_bookkeeper.bookkeeper import Bookkeeper
from picca_bookkeeper.tasker import Tasker

if TYPE_CHECKING:
    from typing import Callable, Optional, Type
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

    bookkeeper = Bookkeeper(
        args.bookkeeper_config, overwrite_config=args.overwrite_config
    )

    logger.info("Adding CoLoRe realisation.")
    colore = bookkeeper.get_colore_tasker(
        wait_for=args.wait_for,
        overwrite=args.overwrite,
        skip_sent=args.skip_sent,
    )

    colore.write_job()
    if not args.only_write:
        colore.send_job()
        logger.info(f"Sent CoLoRe run:\n\t{colore.jobid}")


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

    parser.add_argument(
        "--only-write",
        action="store_true",
        help="Only write scripts, do not send them.",
    )

    parser.add_argument("--wait-for", nargs="+", type=int, default=None, required=False)

    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"],
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    main()
