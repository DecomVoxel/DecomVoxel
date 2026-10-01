from __future__ import annotations

import argparse
import os
import re
import shlex
import shutil
import subprocess
import tarfile
import urllib.request
import json
from typing import Iterable


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_JSON_ENV_VAR = "DECOMVOXEL_CONFIG_JSON"

BLENDER_VERSION_PREFIX = "4.5"
BLENDER_PACKAGE_NAME = "blender-4.5.8-linux-x64"
BLENDER_ARCHIVE_NAME = f"{BLENDER_PACKAGE_NAME}.tar.xz"
BLENDER_DOWNLOAD_URL = "https://www.blender.org/download/release/Blender4.5/blender-4.5.8-linux-x64.tar.xz"
DEFAULT_INSTALL_ROOT = os.path.expanduser(os.environ.get("DECOMVOXEL_BLENDER_INSTALL_ROOT", "/tmp"))
DEFAULT_BLENDER_PATH = os.path.join(DEFAULT_INSTALL_ROOT, BLENDER_PACKAGE_NAME, "blender")
REQUIRED_SYSTEM_PACKAGES = (
	"libxrender1",
	"libxi6",
	"libxkbcommon-x11-0",
	"libsm6",
)


def is_running_in_supported_blender(bpy_module) -> bool:
	if bpy_module is None:
		return False

	app = getattr(bpy_module, "app", None)
	if app is None:
		return False

	version = getattr(app, "version", None)
	if not isinstance(version, tuple) or len(version) < 2:
		return False
	if f"{version[0]}.{version[1]}" != BLENDER_VERSION_PREFIX:
		return False

	binary_path = getattr(app, "binary_path", None)
	if not binary_path:
		return False
	binary_path = os.path.abspath(os.path.expanduser(binary_path))
	if not os.path.isfile(binary_path):
		return False

	return os.path.basename(binary_path).startswith("blender")


def resolve_config_argument(argv: list[str], default_config_path: str) -> str:
	parser = argparse.ArgumentParser(add_help=False)
	parser.add_argument("--config", type=str, default=None)
	parsed, _ = parser.parse_known_args(argv)
	config_path = parsed.config or default_config_path
	return resolve_existing_path(config_path, default_config_path)


def resolve_existing_path(path: str, anchor_path: str) -> str:
	path = os.path.expanduser(path)
	if os.path.isabs(path):
		resolved_path = os.path.abspath(path)
		if os.path.isfile(resolved_path):
			return resolved_path
		raise FileNotFoundError(f"Config file not found: {resolved_path}")

	candidates = [os.path.abspath(path)]
	anchor_dir = os.path.dirname(os.path.abspath(anchor_path))
	current_dir = anchor_dir
	while True:
		candidates.append(os.path.join(current_dir, path))
		parent_dir = os.path.dirname(current_dir)
		if parent_dir == current_dir:
			break
		current_dir = parent_dir

	seen: set[str] = set()
	for candidate in candidates:
		candidate = os.path.abspath(candidate)
		if candidate in seen:
			continue
		seen.add(candidate)
		if os.path.isfile(candidate):
			return candidate

	raise FileNotFoundError(
		f"Config file not found: {path}. Tried: {', '.join(seen)}"
	)


def _load_config(config_path: str) -> dict:
	import yaml

	with open(config_path, "r", encoding="utf-8") as f:
		return yaml.safe_load(f) or {}


def _normalize_passthrough_args(argv: list[str], config_path: str) -> list[str]:
	normalized_args: list[str] = []
	seen_config = False
	skip_next = False

	for index, arg in enumerate(argv):
		if skip_next:
			skip_next = False
			continue
		if arg == "--config":
			seen_config = True
			skip_next = True
			normalized_args.extend(["--config", config_path])
			continue
		if arg.startswith("--config="):
			seen_config = True
			normalized_args.extend(["--config", config_path])
			continue
		normalized_args.append(arg)

	if not seen_config:
		normalized_args.extend(["--config", config_path])

	return normalized_args


def _get_install_root(config: dict) -> str:
	basic = config.get("basic", {})
	install_root = basic.get("blender_installation_path", DEFAULT_INSTALL_ROOT)
	return os.path.abspath(os.path.expanduser(install_root))


def _iter_search_roots() -> Iterable[str]:
	current_dir = SCRIPT_DIR
	while True:
		yield current_dir
		parent_dir = os.path.dirname(current_dir)
		if parent_dir == current_dir:
			break
		current_dir = parent_dir


def _iter_archive_candidates(config: dict, install_root: str) -> Iterable[str]:
	basic = config.get("basic", {})
	candidates = [
		os.environ.get("DECOMVOXEL_BLENDER_ARCHIVE"),
		basic.get("blender_archive_path"),
		os.path.join(install_root, BLENDER_ARCHIVE_NAME),
	]

	for root_dir in _iter_search_roots():
		candidates.extend(
			[
				os.path.join(root_dir, "Blender", "models", BLENDER_ARCHIVE_NAME),
				os.path.join(root_dir, "Blender", BLENDER_ARCHIVE_NAME),
			]
		)

	seen: set[str] = set()
	for candidate in candidates:
		if not candidate:
			continue
		candidate = os.path.abspath(os.path.expanduser(candidate))
		if candidate in seen:
			continue
		seen.add(candidate)
		yield candidate


def _iter_candidate_paths(config: dict) -> Iterable[str]:
	basic = config.get("basic", {})
	install_root = _get_install_root(config)
	candidates = [
		os.environ.get("BLENDER_PATH"),
		basic.get("blender_path"),
		basic.get("blender_binary"),
		os.path.join(install_root, BLENDER_PACKAGE_NAME, "blender"),
		DEFAULT_BLENDER_PATH,
		shutil.which("blender"),
		"/usr/local/bin/blender",
		"/usr/bin/blender",
	]

	seen: set[str] = set()
	for candidate in candidates:
		if not candidate:
			continue
		candidate = os.path.abspath(os.path.expanduser(candidate))
		if candidate in seen:
			continue
		seen.add(candidate)
		yield candidate


def _query_blender_version(blender_path: str) -> str | None:
	try:
		result = subprocess.run(
			[blender_path, "--version"],
			capture_output=True,
			text=True,
			check=False,
			timeout=15,
		)
	except (OSError, subprocess.SubprocessError):
		return None

	output = "\n".join(part for part in (result.stdout, result.stderr) if part)
	for line in output.splitlines():
		line = line.strip()
		if line.startswith("Blender "):
			return line
	return None


def _is_supported_blender(blender_path: str) -> tuple[bool, str | None]:
	if not os.path.isfile(blender_path) or not os.access(blender_path, os.X_OK):
		return False, None

	version_line = _query_blender_version(blender_path)
	if version_line is None:
		return False, None

	match = re.search(r"Blender\s+(\d+\.\d+(?:\.\d+)?)", version_line)
	if match is None:
		return False, version_line

	return match.group(1).startswith(BLENDER_VERSION_PREFIX), version_line


def _maybe_install_system_packages() -> None:
	apt_get = shutil.which("apt-get")
	if apt_get is None:
		return

	if os.geteuid() == 0:
		prefix: list[str] = []
	else:
		sudo = shutil.which("sudo")
		if sudo is None:
			print("[INFO] sudo not found, skipping optional Blender system package installation.")
			return
		prefix = [sudo, "-n"]

	update_cmd = prefix + [apt_get, "update"]
	install_cmd = prefix + [apt_get, "install", "-y", *REQUIRED_SYSTEM_PACKAGES]

	update_result = subprocess.run(update_cmd, check=False, capture_output=True, text=True)
	if update_result.returncode != 0:
		print("[INFO] Could not run apt-get update without interaction, skipping optional package installation.")
		return

	install_result = subprocess.run(install_cmd, check=False, capture_output=True, text=True)
	if install_result.returncode != 0:
		print("[INFO] Optional Blender system package installation failed, continuing with downloaded Blender only.")


def _download_archive(url: str, archive_path: str) -> None:
	os.makedirs(os.path.dirname(archive_path), exist_ok=True)
	print(f"[INFO] Downloading Blender archive to {archive_path}")
	with urllib.request.urlopen(url) as response, open(archive_path, "wb") as f:
		shutil.copyfileobj(response, f)


def _is_valid_archive(archive_path: str) -> bool:
	if not os.path.isfile(archive_path):
		return False
	try:
		return tarfile.is_tarfile(archive_path)
	except OSError:
		return False


def _prepare_archive(config: dict, install_root: str) -> str:
	target_archive_path = os.path.join(install_root, BLENDER_ARCHIVE_NAME)
	for candidate in _iter_archive_candidates(config, install_root):
		if not _is_valid_archive(candidate):
			continue
		if os.path.abspath(candidate) == os.path.abspath(target_archive_path):
			print(f"[INFO] Reusing existing Blender archive: {candidate}")
			return target_archive_path
		os.makedirs(install_root, exist_ok=True)
		print(f"[INFO] Reusing local Blender archive: {candidate}")
		shutil.copy2(candidate, target_archive_path)
		return target_archive_path

	_download_archive(BLENDER_DOWNLOAD_URL, target_archive_path)
	return target_archive_path


def _extract_archive(archive_path: str, install_root: str) -> None:
	print(f"[INFO] Extracting Blender archive into {install_root}")
	with tarfile.open(archive_path, "r:xz") as archive:
		try:
			archive.extractall(path=install_root, filter="data")
		except TypeError:
			archive.extractall(path=install_root)


def _install_blender(config: dict, install_root: str) -> str:
	target_path = os.path.join(install_root, BLENDER_PACKAGE_NAME, "blender")
	if os.path.isfile(target_path):
		return target_path

	os.makedirs(install_root, exist_ok=True)
	_maybe_install_system_packages()
	archive_path = _prepare_archive(config=config, install_root=install_root)
	_extract_archive(archive_path, install_root)
	if not os.path.isfile(target_path):
		raise RuntimeError(f"Blender was extracted but executable was not found: {target_path}")
	os.chmod(target_path, os.stat(target_path).st_mode | 0o111)
	return target_path


def ensure_blender_binary(config: dict) -> str:
	incompatible_versions: list[tuple[str, str]] = []
	for candidate in _iter_candidate_paths(config):
		is_supported, version_line = _is_supported_blender(candidate)
		if is_supported:
			print(f"[INFO] Using Blender at {candidate} ({version_line})")
			return candidate
		if version_line is not None:
			incompatible_versions.append((candidate, version_line))

	if incompatible_versions:
		for path, version_line in incompatible_versions:
			print(f"[INFO] Ignoring incompatible Blender binary: {path} ({version_line})")

	install_root = _get_install_root(config)
	blender_path = _install_blender(config, install_root)
	is_supported, version_line = _is_supported_blender(blender_path)
	if not is_supported:
		raise RuntimeError(f"Installed Blender is not a supported 4.5 build: {blender_path} ({version_line})")
	print(f"[INFO] Installed Blender at {blender_path} ({version_line})")
	return blender_path


def relaunch_in_blender(script_path: str, argv: list[str], default_config_path: str) -> int:
	config_path = resolve_config_argument(argv, default_config_path)
	config = _load_config(config_path)
	blender_path = ensure_blender_binary(config)
	passthrough_args = _normalize_passthrough_args(argv, config_path)
	env = os.environ.copy()
	env[CONFIG_JSON_ENV_VAR] = json.dumps(config)
	command = [
		blender_path,
		"-b",
		"-P",
		os.path.abspath(script_path),
		"--",
		*passthrough_args,
	]
	print(f"[INFO] Launching Blender command: {' '.join(shlex.quote(part) for part in command)}")
	return subprocess.call(command, env=env)