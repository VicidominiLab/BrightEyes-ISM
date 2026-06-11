import h5py
import numpy as np

from brighteyes_ism.dataio import mcs


def _write_legacy_file(path):
    with h5py.File(path, "w") as f:
        f.attrs["data_format_version"] = "0.0.1"
        f.attrs["comment"] = "legacy"
        f.create_dataset("data", data=np.arange(16).reshape(1, 1, 2, 2, 2, 2))

        gui = f.create_group("configurationGUI")
        gui.attrs["range_x"] = 8.0
        gui.attrs["range_y"] = 6.0
        gui.attrs["range_z"] = 1.0
        gui.attrs["timebin_per_pixel"] = 2
        gui.attrs["time_resolution"] = 0.1
        gui.attrs["nx"] = 2
        gui.attrs["ny"] = 2
        gui.attrs["nframe"] = 1
        gui.attrs["nrep"] = 1
        gui.attrs["calib_x"] = 1.0
        gui.attrs["calib_y"] = 1.0
        gui.attrs["calib_z"] = 1.0
        gui.attrs["bitFile"] = "legacy_40M91.lvbitx"

        fpga = f.create_group("configurationFPGA")
        fpga.attrs["DFD_Activate"] = True


def _write_v006_file(path):
    with h5py.File(path, "w") as f:
        f.attrs["data_format_version"] = "0.0.6"
        f.attrs["comment"] = "v006"

        raw = f.create_group("raw")
        raw.create_dataset("spad", data=np.arange(16).reshape(1, 1, 2, 2, 2, 2))
        metadata = raw.create_group("metadata")
        metadata.attrs["range_x_um"] = 8.0
        metadata.attrs["range_y_um"] = 6.0
        metadata.attrs["range_z_um"] = 1.0
        metadata.attrs["offset_x_um"] = 0.0
        metadata.attrs["offset_y_um"] = 0.0
        metadata.attrs["offset_z_um"] = 0.0
        metadata.attrs["pixel_size_x_um"] = 4.0
        metadata.attrs["pixel_size_y_um"] = 3.0
        metadata.attrs["pixel_size_z_um"] = 1.0
        metadata.attrs["time_bins"] = 2
        metadata.attrs["digital_time_bins"] = 2
        metadata.attrs["time_bin_ns"] = 100.0
        metadata.attrs["time_resolution_us"] = 0.1
        metadata.attrs["pixel_dwell_time_us"] = 0.2
        metadata.attrs["pixel_dwell_time_ns"] = 200.0
        metadata.attrs["nx"] = 2
        metadata.attrs["ny"] = 2
        metadata.attrs["nz"] = 1
        metadata.attrs["nrep"] = 1
        metadata.attrs["dfd_active"] = True

        timing = metadata.create_group("acquisition").create_group("timing")
        timing.attrs["laser_frequency_mhz"] = 80.0
        timing.attrs["dfd_time_bins"] = 128
        timing.attrs["dfd_trigger_selector"] = 1
        timing.attrs["dfd_laser_sync_debug"] = False


def test_load_keeps_legacy_layout_for_old_versions(tmp_path):
    path = tmp_path / "legacy.h5"
    _write_legacy_file(path)

    data, meta = mcs.load(path)

    assert data.shape == (1, 1, 2, 2, 2, 2)
    assert isinstance(meta, mcs.metadata)
    assert meta.version == "0.0.1"
    assert meta.nbin == 2
    assert meta.dfd_freq == 40
    assert meta.dfd_nbins == 91


def test_load_uses_raw_spad_and_normalized_metadata_for_v006(tmp_path):
    path = tmp_path / "v006.h5"
    _write_v006_file(path)

    data, meta = mcs.load(path)

    assert data.shape == (1, 1, 2, 2, 2, 2)
    assert isinstance(meta, mcs.metadata_v006)
    assert meta.version == "0.0.6"
    assert meta.comment == "v006"
    assert meta.dx == 4.0
    assert meta.dy == 3.0
    assert meta.pxdwelltime == 0.2
    assert meta.dfd_freq == 80.0
    assert meta.dfd_nbins == 128


def test_v006_data_alias_for_aux_channels(tmp_path):
    path = tmp_path / "v006_aux.h5"
    with h5py.File(path, "w") as f:
        f.attrs["data_format_version"] = "0.0.6"
        f.attrs["comment"] = "v006"
        raw = f.create_group("raw")
        raw.create_dataset("spad", data=np.zeros((1, 1, 1, 1, 1, 1)))
        raw.create_dataset("aux", data=np.ones((1, 1, 1, 1, 1, 2)))
        metadata = raw.create_group("metadata")
        metadata.attrs["range_x_um"] = 1.0
        metadata.attrs["range_y_um"] = 1.0
        metadata.attrs["range_z_um"] = 1.0
        metadata.attrs["offset_x_um"] = 0.0
        metadata.attrs["offset_y_um"] = 0.0
        metadata.attrs["offset_z_um"] = 0.0
        metadata.attrs["pixel_size_x_um"] = 1.0
        metadata.attrs["pixel_size_y_um"] = 1.0
        metadata.attrs["pixel_size_z_um"] = 1.0
        metadata.attrs["time_bins"] = 1
        metadata.attrs["time_bin_ns"] = 100.0
        metadata.attrs["time_resolution_us"] = 0.1
        metadata.attrs["pixel_dwell_time_us"] = 0.1
        metadata.attrs["pixel_dwell_time_ns"] = 100.0
        metadata.attrs["nx"] = 1
        metadata.attrs["ny"] = 1
        metadata.attrs["nz"] = 1
        metadata.attrs["nrep"] = 1
        metadata.attrs["dfd_active"] = True

        timing = metadata.create_group("acquisition").create_group("timing")
        timing.attrs["laser_frequency_mhz"] = 80.0
        timing.attrs["dfd_time_bins"] = 1
        timing.attrs["dfd_trigger_selector"] = 1
        timing.attrs["dfd_laser_sync_debug"] = False

    data, _ = mcs.load(path, key="data_channels_extra")

    assert data.shape == (1, 1, 1, 1, 1, 2)
    assert np.all(data == 1)
