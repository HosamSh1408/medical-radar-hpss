# External Benchmark Dataset: Gleichauf et al. (2024)

**Citation:** Gleichauf, J., Herrmann, S., Niebler, C., & Kölpin, A. (2024). *A Time-of-Flight and Radar Dataset of a neonatal Thorax Simulator with synchronized Reference Sensor Signals for respiratory Rate Detection*. *Scientific Data*, 11, 110. DOI: [10.1038/s41597-024-02946-5](https://doi.org/10.1038/s41597-024-02946-5)  
**Figshare Collection:** DOI `10.6084/m9.figshare.c.6809370.v1` (CC BY 4.0)  
**Role in Thesis (`L4` Claim Gate):** Untouched external 24.2 GHz CW radar + physical displacement reference benchmark (`WP1` Jacobi–Anger / I/Q calibration & zero-leakage generalization).

---

## Downloaded Subset Summary

* **Included (`49` files, `3.577 GiB` / `3.8 GB` on disk):**
  * **20 scenarios** (`05, 10, 20, 30, 40, 45, 50, 55, 60, 80` BPM × `{normal, deep}`), each containing:
    * `iq_values_<scenario>.csv`: 24.2 GHz InnoSenT iSYS-4001 CW radar baseband (~2 kHz):  
      `%time,field.header.seq,field.header.stamp,field.header.frame_id,field.sample_id,field.I,field.Q`
    * `laser_<scenario>.csv`: Micro-Epsilon optoCONTROL 2520 laser micrometer displacement reference (~2.5 kHz, headerless):  
      `<unix_timestamp_s>,<displacement_mm>`
  * `scripts/` (`9` files): Official MATLAB/Octave, ROS, and shell conversion utilities from Figshare article `24056274`.
* **Skipped (`ToF_*.zip`):** Nested 3D Time-of-Flight camera archives (~30 GB total) were skipped via HTTP Range extraction. All original download URLs and Figshare article DOIs remain logged in `manifest.json`, and SHA-256 checksums of every extracted local file are stored in `extracted_manifest.json`.
