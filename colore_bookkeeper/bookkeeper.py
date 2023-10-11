from __future__ import annotations

import configparser
import copy
import filecmp
import logging
import libconf
import shutil
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import yaml
from importlib_resources import files
from picca_bookkeeper.bookkeeper import (
    Bookkeeper as PiccaBookkeeper,
    PathBuilder as PiccaPathBuilder,
)
from picca_bookkeeper.dict_utils import DictUtils
from picca_bookkeeper.tasker import ChainedTasker, DummyTasker, Tasker, get_Tasker
from yaml import SafeDumper

from colore_bookkeeper import resources

if TYPE_CHECKING:
    from typing import Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# This converts Nones in dict into empty fields in yaml.
SafeDumper.add_representer(
    type(None),
    lambda dumper, value: dumper.represent_scalar("tag:yaml.org,2002:null", ""),
)


class Bookkeeper(PiccaBookkeeper):
    """Class to generate Tasker objects which can be used to run different Lyman-alpha mock
    building steps.
    """

    label: Optional[str]
    generate_slurm_header_extra_args: Callable = (
        PiccaBookkeeper.generate_slurm_header_extra_args
    )
    generate_extra_args: Callable = PiccaBookkeeper.generate_extra_args

    def __init__(
        self,
        config_path: str | Path,
        overwrite_config: bool = False,
        read_mode: bool = False,
    ):
        config_path = Path(config_path)
        if not config_path.is_file():
            if (config_path / "configs/bookkeeper_config.yaml").is_file():
                config_path = config_path / "configs/bookkeeper_config.yaml"
            else:
                raise FileNotFoundError("Config file couldn't be found", config_path)

        with open(config_path) as file:
            self.config = yaml.safe_load(file)

        self.paths = PathBuilder(self.config)

        self.colore = None
        self.lyacolore = None
        self.quickquasars = None

        if self.config.get("quickquasars") is not None:
            self.quickquasars = self.config.get("quickquasars")
            config_type = "quickquasars"
        if self.config.get("LyaCoLoRe") is not None:
            self.lyacolore = self.config.get("LyaCoLoRe")
            config_type = "LyaCoLoRe"
        if self.config.get("CoLoRe") is not None:
            self.colore = self.config.get("CoLoRe")
            config_type = "CoLoRe"

        if config_type == "quickquasars":
            # In this case, LyaCoLoRe is not defined in the config file
            # and therefore we should search for it
            with open(self.paths.lyacolore_config_file, "r") as f:
                lyacolore_config = yaml.safe_load(f)
            self.lyacolore = lyacolore_config["LyaCoLoRe"]
            self.config["LyaCoLoRe"] = self.lyacolore

        if config_type in ("quickquasars", "LyaCoLoRe"):
            # In this case, CoLoRe is not defined in the config file
            # and therefore we should search for it
            with open(self.paths.colore_config_file, "r") as f:
                colore_config = yaml.safe_load(f)
            self.colore = colore_config["CoLoRe"]
            self.config["CoLoRe"] = self.colore

        self.paths = PathBuilder(self.config)

        if read_mode:
            # Next steps imply writing on bookkeeper's destination
            # for read_mode we can finish here.
            return

        # Check directory structure
        self.paths.check_colore_directories()
        if self.lyacolore is not None:
            self.paths.check_lyacolore_directories()
        if self.quickquasars is not None:
            self.paths.check_quickquasars_directories()

        # Copy bookkeeper configuration into destination
        if config_type == "CoLoRe":
            config_colore = copy.deepcopy(self.config)
            config_colore.pop("correlations", None)
            config_colore.pop("fits", None)

            if not self.paths.colore_config_file.is_file():
                self.write_bookkeeper(config_colore, self.paths.colore_config_file)
            elif filecmp.cmp(self.paths.colore_config_file, config_path):
                # If files are the same we can continue
                pass
            elif overwrite_config:
                # If we want to directly overwrite the config file in destination
                self.write_bookkeeper(config_colore, self.paths.colore_config_file)
            else:
                comparison = PathBuilder.compare_config_files(
                    config_path, self.paths.colore_config_file, "CoLoRe"
                )
                if comparison == dict():
                    # They are the same
                    self.write_bookkeeper(config_colore, self.paths.colore_config_file)
                else:
                    raise ValueError(
                        "CoLoRe section of config file should match CoLoRe "
                        "section from file already in the bookkeeper. "
                        "Unmatching items:\n\n"
                        f"{DictUtils.print_dict(comparison)}\n\n"
                        f"Remove config file to overwrite {self.paths.colore_config_file}"
                    )
        if self.lyacolore is not None and config_type != "quickquasars":
            config_lyacolore = copy.deepcopy(self.config)

            config_lyacolore["LyaCoLoRe"][
                "CoLoRe run name"
            ] = self.paths.colore_run_name
            config_lyacolore.pop("CoLoRe", None)
            config_lyacolore.pop("quickquasars", None)

            if not self.paths.lyacolore_config_file.is_file():
                self.write_bookkeeper(
                    config_lyacolore, self.paths.lyacolore_config_file
                )
            elif filecmp.cmp(self.paths.lyacolore_config_file, config_path):
                # If files are the same we can continue
                pass
            elif overwrite_config:
                # If we want to directly overwrite the config file in destination
                self.write_bookkeeper(
                    config_lyacolore, self.paths.lyacolore_config_file
                )
            else:
                comparison = PathBuilder.compare_config_files(
                    config_path,
                    self.paths.lyacolore_config_file,
                    "LyaCoLoRe",
                    "CoLoRe run name",
                )
                if comparison == dict():
                    # They are the same
                    self.write_bookkeeper(
                        config_lyacolore, self.paths.lyacolore_config_file
                    )
                else:
                    raise ValueError(
                        "LyaCoLoRe section of config file should match section from "
                        "file already in the bookkeeper. "
                        "Unmatching items:\n\n"
                        f"{DictUtils.print_dict(comparison)}\n\n"
                        f"Remove config file to overwrite {self.paths.delta_config_file}"
                    )

        if self.quickquasars is not None:
            config_quickquasars = copy.deepcopy(self.config)

            config_quickquasars["quickquasars"][
                "CoLoRe run name"
            ] = self.paths.colore_run_name
            config_quickquasars["quickquasars"][
                "LyaCoLoRe run name"
            ] = self.paths.lyacolore_run_name

            config_quickquasars.pop("CoLoRe", None)
            config_quickquasars.pop("LyaCoLoRe", None)

            if not self.paths.quickquasars_config_file.is_file():
                self.write_bookkeeper(
                    config_quickquasars, self.paths.quickquasars_config_file
                )
            elif filecmp.cmp(self.paths.quickquasars_config_file, config_path):
                # If files are the same we can continue
                pass
            elif overwrite_config:
                self.write_bookkeeper(
                    config_quickquasars, self.paths.quickquasars_config_file
                )
            else:
                comparison = PathBuilder.compare_config_files(
                    config_path,
                    self.paths.quickquasars_config_file,
                    "quickquasars",
                    ["CoLoRe run name", "LyaCoLoRe run name"],
                )
                if comparison == dict():
                    self.write_bookkeeper(
                        config_quickquasars, self.paths.quickquasars_config_file
                    )
                else:
                    raise ValueError(
                        "quickquasars section of config file should match section from "
                        "file already in the bookkeeper. "
                        "Unmatching items:\n\n"
                        f"{DictUtils.print_dict(comparison)}\n\n"
                        f"Remove config file to overwrite {self.paths.quickquasars_config_file}"
                    )

        # Read defaults and check if they have changed.
        defaults_file = files(resources).joinpath(
            "default_configs/" + str(self.config["general"]["defaults"]) + ".yaml"
        )
        if not defaults_file.is_file():
            raise ValueError("Invalid defaults file. ", defaults_file)

        self.defaults = yaml.safe_load(defaults_file.read_text())
        self.defaults_diff = dict()

        if self.paths.defaults_file.is_file():
            self.defaults_diff = PathBuilder.compare_config_files(
                self.paths.defaults_file,
                defaults_file,
            )
        else:
            self.defaults_diff = {}
            self.write_bookkeeper(self.defaults, self.paths.defaults_file)
        
    @staticmethod
    def write_bookkeeper(config: Dict, file: Path | str) -> None:
        """Method to write bookkeeper yaml file to file

        Args:
            config: Dict to store as yaml file.
            file: path where to store the bookkeeper.
        """
        correct_order = {
            "general": ["conda environment", "system", "slurm args", "defaults"],
            "data": ["bookkeeper dir",],
            "CoLoRe": [
                "run name",
                "CoLoRe directory",
                "OMP_THREADS",
                "extra args",
                "slurm args",
            ],
            "LyaCoLoRe": [
                "run name",
                "CoLoRe run name",
                "extra args",
                "slurm args",
            ],
            "quickquasars": [
                "run name",
                "LyaCoLoRe run name",
                "CoLoRe run name",
                "extra args",
                "slurm args",
            ]
        }

        try:
            config = dict(
                sorted(config.items(), key=lambda s: list(correct_order).index(s[0]))
            )

            for key, value in config.items():
                config[key] = dict(
                    sorted(value.items(), key=lambda s: correct_order[key].index(s[0]))
                )
        except ValueError as e:
            raise ValueError(f"Invalid item in config file").with_traceback(
                e.__traceback__
            )

        with open(file, "w") as f:
            yaml.safe_dump(config, f, sort_keys=False)

    def write_cfg(self, config: Dict, file: Path | str) -> None:
        """Safely save a dictionary into an .cfg file"""
        with open(file, "w") as f:
            libconf.dump(config, f)

    def get_colore_tasker(
        self,
        system: Optional[str] = None,
        wait_for: Optional[Tasker | ChainedTasker | int | List[int]] = None,
        slurm_header_extra_args: Dict = dict(),
        extra_args: Dict = dict(),
        overwrite: bool = False,
        skip_sent: bool = True,
    ) -> Tasker:
        """Method to get a Tasker object to run CoLoRe.

        Args:
            system: Shell to use for job. 'slurm_cori' to use slurm scripts on
                cori, 'slurm_perlmutter' to use slurm scripts on perlmutter,
                'bash' to run it in login nodes or computer shell.
                Default: None, read from config file.
            wait_for: In NERSC, wait for a given job to finish before running
                the current one. Could be a  Tasker object or a slurm jobid
                (int). (Default: None, won't wait for anything).
            slurm_header_extra_args: Change slurm header default options if
                needed (time, qos, etc...). Use a dictionary with the format
                {'option_name': 'option_value'}.
            extra_args: Set extra options for picca delta extraction.
                The format should be a dict of dicts: wanting to change
                "num masks" in "masks" section one should pass
            overwrite: Overwrite files in destination.
            skip_sent: Skip this and return a DummyTasker if the run
                was already sent before.
        """
        if self.defaults_diff != {}:
            raise ValueError(
                "Default values changed since last run of the "
                f"bookkeeper. Remove the file:\n\n {self.paths.defaults_file} "
                "\n\n to be able to write jobs (with the new default "
                f"values\). Defaults diff:\n\n"
                f"{DictUtils.print_dict(self.defaults_diff)}"
            )

        job_name = "CoLoRe"

        updated_system = self.generate_system_arg(system)

        # Check if output already there
        if self.check_existing_output_file(
            self.paths.colore_jobid_file(),
            job_name,
            skip_sent,
            overwrite,
            updated_system,
        ):
            return DummyTasker()

        command = self.config["CoLoRe"]["CoLoRe directory"]

        updated_extra_args = self.generate_extra_args(
            config=self.config,
            default_config=self.defaults,
            section="CoLoRe",
            command="CoLoRe",
            extra_args=dict(),
        )

        updated_slurm_header_extra_args = self.generate_slurm_header_extra_args(
            config=self.config,
            default_config=self.defaults,
            section="CoLoRe",
            command="CoLoRe",
            slurm_args=dict(),
        )

        config_file = self.paths.run_path / f"configs/param_config.ini"

        param_config_dict = DictUtils.merge_dicts(
            {"global": {"prefix_out": str(self.paths.run_path / "results" / "out")}},
            updated_extra_args,
        )
        self.write_cfg(param_config_dict, config_file)
        
        slurm_header_args = {
            "job-name": job_name,
            "output": str(self.paths.run_path / f"logs/{job_name}-%j.out"),
            "error": str(self.paths.run_path / f"logs/{job_name}-%j.err"),
        }

        slurm_header_args = DictUtils.merge_dicts(
            slurm_header_args, updated_slurm_header_extra_args,
        )

        return get_Tasker(updated_system)(
            command=command,
            command_args={"": str(config_file.resolve())},
            slurm_header_args=slurm_header_args,
            environment=self.config["general"]["conda environment"],
            run_file=self.paths.run_path / f"scripts/run_{job_name}.sh",
            jobid_log_file=self.paths.run_path / f"logs/jobids.log",
            wait_for = wait_for,
            out_file=self.paths.colore_jobid_file(),
            force_OMP_threads=self.config["CoLoRe"].get("OMP_THREADS", 1),
        )


class PathBuilder(PiccaPathBuilder):
    """Class to define paths following the bookkeeper convention."""

    def __init__(self, config: Dict):
        self.config = config

    @property
    def colore_run_name(self) -> str:
        # Consistency for CoLoRe run name:
        colore_names = set()
        colore_names.add(
            self.config.get("quickquasars", dict()).get("CoLoRe run name", "")
        )
        colore_names.add(
            self.config.get("LyaCoLoRe", dict()).get("CoLoRe run name", "")
        )
        colore_names.add(self.config.get("CoLoRe", dict()).get("run name", ""))

        colore_names.discard("")
        colore_names.discard(None)

        if len(colore_names) != 1:
            raise ValueError("Incompatible run names for CoLoRe: ", colore_names)
        else:
            return list(colore_names)[0]

    @property
    def lyacolore_run_name(self) -> str:
        # Consistency for LyaCoLoRe run name:
        lyacolore_names = set()
        lyacolore_names.add(
            self.config.get("quickquasars", dict()).get("LyaCoLoRe run name", "")
        )
        lyacolore_names.add(self.config.get("LyaCoLoRe", dict()).get("run name", ""))

        lyacolore_names.discard("")
        lyacolore_names.discard(None)

        if len(lyacolore_names) != 1:
            raise ValueError("Incompatible run names for LyaCoLoRe: ", lyacolore_names)
        else:
            return list(lyacolore_names)[0]

    @property
    def quickquasars_run_name(self) -> str:
        return self.config.get("quickquasars", dict()).get("run name", None)

    @property
    def run_path(self) -> Path:
        """Give full path to bookkeeper run"""
        bookkeeper_dir = Path(self.config["data"]["bookkeeper dir"])
        defaults_name = self.config["general"]["defaults"]

        return bookkeeper_dir / defaults_name / self.colore_run_name

    @property
    def lyacolore_path(self) -> Path:
        """Give full path to lyacolore run"""
        return self.run_path / "LyaCoLoRe" / self.lyacolore_run_name

    @property
    def quickquasars_path(self) -> Path:
        """Give full path to quickquasars run"""
        return self.lyacolore_path / "quickquasars" / self.quickquasars_run_name

    @property
    def colore_config_file(self) -> Path:
        """Path to configuration file for CoLoRe inside bookkeeper."""
        return self.run_path / "configs" / "bookkeeper_config.yaml"

    @property
    def lyacolore_config_file(self) -> Path:
        """Path to configuration file for LyaCoLoRe inside bookkeeper."""
        return self.lyacolore_path / "configs" / "bookkeeper_config.yaml"

    @property
    def quickquasars_config_file(self) -> Path:
        """Path to configuration file for quickquasars inside bookkeeper."""
        return self.quickquasars_path / "configs" / "bookkeeper_config.yaml"

    @property
    def defaults_file(self) -> Path:
        """Location of the defaults file inside the bookkeeper"""
        return self.colore_config_file.parent / "defaults.yaml"

    def check_colore_directories(self) -> None:
        """Method to create basic directories"""
        for folder in ("scripts", "results", "logs", "configs"):
            (self.run_path / folder).mkdir(exist_ok=True, parents=True)

    def check_lyacolore_directories(self) -> None:
        """Method to create basic LyaCoLoRe directories"""
        for folder in ("scripts", "results", "logs", "configs"):
            (self.lyacolore_path / folder).mkdir(exist_ok=True, parents=True)

    def check_quickquasars_directories(self) -> None:
        """Method to create basic LyaCoLoRe directories"""
        for folder in ("scripts", "results", "logs", "configs"):
            (self.quickquasars_path / folder).mkdir(exist_ok=True, parents=True)

    def colore_jobid_file(self) -> Path:
        """Jobid file to keep track of CoLoRe run status"""
        return self.run_path / "results" / "jobid.out"

    def copied_colore_files(self) -> Path | None:
        """
        Method to get path to CoLoRe files if it appears
        in the bookkeeper config file (instead of computing it)
        """
        files = self.config["CoLoRe"].get("copy files", "")

        if files not in ("", None):
            assert files.is_dir()
            return files
        else:
            return None
