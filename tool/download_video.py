
import fiftyone as fo
import fiftyone.zoo as foz

dataset = foz.load_zoo_dataset(
    "kinetics-700-2020",
    split="validation",
    classes=["grooming cat", "grooming dog"],
    max_samples=10,
)

session = fo.launch_app(dataset, port=5151)