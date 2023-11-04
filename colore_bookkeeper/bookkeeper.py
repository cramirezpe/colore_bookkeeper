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

        self.read_mode = read_mode
        self.overwrite_config = overwrite_config

        with open(config_path) as file:
            self.config = yaml.safe_load(file)

        self.paths = PathBuilder(self.config)

        self.colore = None
        self.lyacolore = None
        self.quickquasars = None
        self.corrfunc = None

        if self.config.get("QuickQuasars") is not None:
            self.quickquasars = self.config.get("quickquasars")
        if self.config.get("LyaCoLoRe") is not None:
            self.lyacolore = self.config.get("LyaCoLoRe")
        if self.config.get("Corrfunc") is not None:
            self.corrfunc = self.config.get("Corrfunc")
        if self.config.get("CoLoRe") is not None:
            self.colore = self.config.get("CoLoRe")

        if self.quickquasars is not None:
            if self.lyacolore is None:
                self.lyacolore = yaml.safe_load(self.paths.lyacolore_config_file.read_text())["LyaCoLoRe"]
                self.config["LyaCoLoRe"] = self.lyacolore

            if not self.read_mode:
                self.paths.check_quickquasars_directories()
                self.check_existing_config("QuickQuasars", self.paths.quickquasars_config_file)

        if self.lyacolore is not None:
            if self.colore is None:
                self.colore = yaml.safe_load(self.paths.run_path.read_text())
                self.config["CoLoRe"] = self.colore

            if not self.read_mode:
                self.paths.check_lyacolore_directories()
                self.check_existing_config("LyaCoLoRe", self.paths.lyacolore_config_file)


        if self.corrfunc is not None:
            if self.colore is None:
                self.colore = yaml.safe_load(self.paths.run_path.read_text())
                self.config["CoLoRe"] = self.colore

            if not self.read_mode:
                self.paths.check_corrfunc_directories()
                self.check_existing_config("Corrfunc", self.paths.corrf_config_file)

        if self.colore is not None:
            if not self.read_mode:
                self.paths.check_colore_directories()
                self.check_existing_config("CoLoRe", self.paths.colore_config_file)

        self.paths = PathBuilder(self.config)
       
    @staticmethod
    def write_bookkeeper(config: Dict, file: Path | str) -> None:
        """Method to write bookkeeper yaml file to file

        Args:
            config: Dict to store as yaml file.
            file: path where to store the bookkeeper.
        """
        correct_order = {
            "general": ["conda environment", "system", "slurm args"],
            "data": ["bookkeeper dir",],
            "CoLoRe": [
                "run name",
                "CoLoRe directory",
                "OMP_THREADS",
                "extra args",
                "slurm args",
            ],
            "Corrfunc": [
                "run name",
                "source",
                "CoLoRe run name",
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
            overwrite: Overwrite files in destination.
            skip_sent: Skip this and return a DummyTasker if the run
                was already sent before.
        """
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
            section="CoLoRe",
            command="CoLoRe",
        )

        updated_slurm_header_extra_args = self.generate_slurm_header_extra_args(
            config=self.config,
            section="CoLoRe",
            command="CoLoRe",
        )

        config_file = self.paths.run_path / f"configs/param_config.cfg"

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
            out_files=[self.paths.colore_jobid_file(),],
            force_OMP_threads=self.config["CoLoRe"].get("OMP_THREADS", 1),
        )

    def get_lyacolore_tasker(
        self,
        system: Optional[str] = None,
        wait_for: Optional[Tasker | ChainedTasker | int | List[int]] = None,
        overwrite: bool = False,
        skip_sent: bool = True,
    ) -> Tasker:
        """Method to get a Tasker object to run LyaCoLoRe.

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
            overwrite: Overwrite files in destination.
            skip_sent: Skip this and return a DummyTasker if the run
                was already sent before.
        """       
        job_name = "LyaCoLoRe"

        updated_system = self.generate_system_arg(system)

        # Check if output already there

    def get_corrf_tasker(
        self,
        system: Optional[str] = None,
        wait_for: Optional[Tasker | ChainedTasker | int | List[int]] = None,
        overwrite: bool = False,
        skip_sent: bool = True,
    ) -> Tasker:
        """Method to get a Tasker object to run LyaCoLoRe.

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
            overwrite: Overwrite files in destination.
            skip_sent: Skip this and return a DummyTasker if the run
                was already sent before.
        """        
        job_name = "modules_corrf"

        updated_system = self.generate_system_arg(system)
        
        command = "CoLoRe_corrf_run_correlations"

        files = "DD", "DR", "RR"
        existing_files = set()
        out_files = set()
        for file in files:
            output_filename = self.paths.corrf_path / "results" / f"{file}.dat"
            # Check if output already there
            if self.check_existing_output_file(
                output_filename,
                job_name,
                skip_sent,
                overwrite,
                updated_system,
            ):
                existing_files.add(file)
                continue
            else:
                copy_file = self.paths.copied_corrf_file(
                    file,
                )

                if copy_file is not None:
                    output_filename = self.paths.corrf_path / "results" / f"{file}.dat"
                    output_filename.unlink(missing_ok=True)
                    output_filename.parent.mkdir(exist_ok=True, parents=True)
                    output_filename.symlink_to(copy_file)

                    exiting_files.add(file)
            
            out_files.append(output_filename)

        if len(existing_files) == 3:
            return DummyTasker()

        updated_extra_args = self.generate_extra_args(
            config=self.config,
            section="Corrfunc",
            command=command,
        )

        updated_slurm_header_extra_args = self.generate_slurm_header_extra_args(
            config=self.config,
            section="Corrfunc",
            command=command,
        )

        slurm_header_args = {
            "job-name": job_name,
            "output": str(self.paths.corrf_path / f"logs/{job_name}-%j.out"),
            "error": str(self.paths.corrf_path / f"logs/{job_name}-%j.err"),
        }

        slurm_header_args = DictUtils.merge_dicts(
            slurm_header_args, updated_slurm_header_extra_args
        )

        source = self.config["Corrfunc"]["source"]
        args = {
            "data": str(self.paths.run_path / "results" / f"out_srcs_s{source}_*"),
            "log-level": "DEBUG",
            "data-format": "CoLoRe",
            "out-dir": self.paths.corrf_path / "results",
        }

        args = DictUtils.merge_dicts(args, updated_extra_args)

        self.paths.corrf_jobid_file().parent.mkdir(exist_ok=True, parents=True)

        return get_Tasker(updated_system)(
            command=command, 
            command_args=args,
            slurm_header_args=slurm_header_args,
            environment=self.config["general"]["conda environment"],
            run_file=self.paths.corrf_path / f"scripts/run_{job_name}.sh",
            jobid_log_file=self.paths.corrf_path / f"logs/jobids.log",
            wait_for=wait_for,
            in_files=[self.paths.colore_jobid_file(),],
            out_files=out_files,
        )

        
    # def get_corrf_tasker_single_pixel(
    #     self,
    #     pixel: int,
    #     system: Optional[str] = None,
    #     wait_for: Optional[Tasker | ChainedTasker | int | List[int]] = None,
    #     overwrite: bool = False,
    #     skip_sent: bool = True,
    # ) -> Tasker:
    #     """Method to get a Tasker object to run LyaCoLoRe.

    #     Args:
    #         pixel: correspondent healpix pixel.
    #         system: Shell to use for job. 'slurm_cori' to use slurm scripts on
    #             cori, 'slurm_perlmutter' to use slurm scripts on perlmutter,
    #             'bash' to run it in login nodes or computer shell.
    #             Default: None, read from config file.
    #         wait_for: In NERSC, wait for a given job to finish before running
    #             the current one. Could be a  Tasker object or a slurm jobid
    #             (int). (Default: None, won't wait for anything).
    #         slurm_header_extra_args: Change slurm header default options if
    #             needed (time, qos, etc...). Use a dictionary with the format
    #             {'option_name': 'option_value'}.
    #         overwrite: Overwrite files in destination.
    #         skip_sent: Skip this and return a DummyTasker if the run
    #             was already sent before.
    #     """
    #     job_name = "modules_corrf"

    #     updated_system = self.generate_extra_args(system)

    #     # Check if output already there,
    #     if self.check_existing_jobid_file(
    #         self.paths.corrf_jobid_file(),
    #         job_name,
    #         skip_sent,
    #         overwrite,
    #         updated_system,
    #     ):
    #         return DummyTasker()

    #     updated_extra_args = self.generate_extra_args(
    #         config=self.config,
    #         section="Corrf",
    #         command="corrf", # update command here
    #     )

    #     updated_slurm_header_extra_args = self.generate_slurm_header_extra_args(
    #         config=self.config,
    #         section="Corrf",
    #         command="Corrf",
    #     )

    #     config_file = self.paths.corrf_path / f"configs/{pixel}"

        

class PathBuilder(PiccaPathBuilder):
    """Class to define paths following the bookkeeper convention."""

    def __init__(self, config: Dict):
        self.config = config

    def copied_corrf_file(file: str) -> Path:
        parent = self.config["Corrf"].get("copy results", None)

        if parent is not None:
            parent = Path(parent)
            result_file = parent / file + ".dat"

            if result_file.is_file():
                logger.info(f"Corrf {file}: Using from file:\n\t{str(result_file)}")
                return result_file
            else:
                logger.info(
                    f"Corrf {file}: No file provided to copy, it will be computed."
                )
                return None
        
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
        colore_names.add(
            self.config.get("Corrfunc", dict()).get("CoLoRe run name", "")
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
        return self.config.get("QuickQuasars", dict()).get("run name", None)
    
    @property
    def corrf_run_name(self) -> str:
        return self.config.get("Corrfunc", dict()).get("run name", None)

    @property
    def run_path(self) -> Path:
        """Give full path to bookkeeper run"""
        bookkeeper_dir = Path(self.config["data"]["bookkeeper dir"])

        return bookkeeper_dir / self.colore_run_name

    @property
    def lyacolore_path(self) -> Path:
        """Give full path to lyacolore run"""
        return self.run_path / "LyaCoLoRe" / self.lyacolore_run_name

    @property
    def corrf_path(self) -> Path:
        """Give full path to corrf run"""
        return self.run_path / "Corrf" / self.corrf_run_name

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
    def corrf_config_file(self) -> Path:
        """Path to configuration file for quickquasars inside bookkeeper."""
        return self.corrf_path / "configs" / "bookkeeper_config.yaml"

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
    
    def check_corrfunc_directories(self) -> None:
        """Method to create basic Corrfunc directories"""
        for folder in ("scripts", "results", "logs", "configs"):
            (self.corrf_path / folder).mkdir(exist_ok=True, parents=True)

    def colore_jobid_file(self) -> Path:
        """Jobid file to keep track of CoLoRe run status"""
        return self.run_path / "results" / "jobid.out"
        
    def corrf_jobid_file(self) -> Path:
        """Jobid file to keep track of corrf status"""
        return self.corrf_path / "results" / "jobid.out"

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
