## Project Structure

```text
cloud-removal/
├── .gitignore
├── data/
│   ├── cropped/
│   ├── patches/
│   ├── raw/
│   └── training_pairs/
└── src/
    ├── cloud_mask.py
    ├── crop_aoi.py
    ├── data_utils.py
    ├── debug_cloud_threshold.py
    ├── extract_patches.py
    ├── pick_aoi.py
    ├── visualize_cloud_mask.py
    └──model.py
```
## Dataset

The dataset used in this project was obtained from the **Bhoonidhi ISRO Portal**.

To respect the portal's usage terms and avoid any uncertainty regarding redistribution during the course of the hackathon, the dataset is **not included** in this
repository. If you wish to reproduce this work, please obtain the data directly from the official Bhoonidhi ISRO Portal.

After downloading the required imagery, organize the files according to the project structure shown above.
