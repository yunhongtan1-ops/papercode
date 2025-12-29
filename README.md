
---

## Dataset

### iSOOD Dataset (Original)

All experiments in this work are conducted on the **iSOOD dataset**, a publicly released
benchmark jointly developed by **Tsinghua University** and the **Basin Environmental Research Center
of the Ministry of Ecology and Environment**.

The dataset contains **10,481 field images** acquired using unmanned aerial vehicles and handheld
cameras across the **Yangtze and Yellow River basins**, covering diverse natural–artificial
composite scenes such as river channels, shorelines, vegetation, buildings, and sewage outfalls.

- **Original dataset download (Zenodo):**  
  https://zenodo.org/records/10903574

> **Note:** Due to licensing and storage constraints, the dataset itself is **not included**
> in this repository and should be downloaded separately from the official source above.

---

## Small-Object Subset Construction

To specifically evaluate detection robustness under small-object conditions,
a dedicated small-object subset is constructed from the original iSOOD dataset.

### Definition of Small Objects
An image is retained if it contains **at least one sewage outfall instance**
whose bounding-box area occupies **less than 0.5% of the image area**.
Under the YOLO annotation format, this criterion is implemented as:

