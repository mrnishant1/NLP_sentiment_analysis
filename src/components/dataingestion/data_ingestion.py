import dataclasses
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from config import load_config,PROJECT_ROOT

cfg = load_config()

@dataclasses.dataclass
class DataIngestionConfig:
    raw_dataset: Path = Path(cfg["paths"]["raw_data"])


@dataclasses.dataclass
class CreateDataArtifact:
    project_root: Path = PROJECT_ROOT
    raw_data_dir: Path = project_root / "raw_data"
    artifacts: Path = project_root / "artifacts"
    
    raw_dataset: Path = project_root / cfg["paths"]["raw_data"] # "./raw_data/data.csv"

    train_dataset: Path = project_root / cfg["paths"]["train_data"] # "./artifacts/train.csv"
    test_dataset: Path = project_root / cfg["paths"]["test_data"] # "./artifacts/test.csv"
    validation_dataset: Path = project_root / "artifacts/valid.csv"


class DataIngestion:
    """Validate source data and create reproducible train/test artifacts.

    The source CSV is never modified. Only row-level validation that does not
    learn from data is performed before splitting; learned transforms must be
    fit using the training split only.
    """

    TEXT_COLUMN = "clean_comment"
    LABEL_COLUMN = "category"
    VALID_LABELS = {-1, 0, 1}
    
    def __init__(self):
        self.DataIngestionConfig = DataIngestionConfig()
        self.CreateDataArtifact = CreateDataArtifact()
        self.SEED = cfg['seed']

    def initiate_data_ingestion(self):
        raw_dataset_path = self.DataIngestionConfig.raw_dataset
        
        train_dataset_path = self.CreateDataArtifact.train_dataset
        test_dataset_path = self.CreateDataArtifact.test_dataset
        valid_dataset_path = self.CreateDataArtifact.validation_dataset
        validated_data_path = cfg["paths"]["validated_data"] 
        
        
        
        self.CreateDataArtifact.raw_data_dir.mkdir(parents=True, exist_ok=True)
        self.CreateDataArtifact.artifacts.mkdir(parents=True, exist_ok=True)

        # Keep ``raw_dataset_path`` immutable. Validated data is written only
        # to the generated train/test artifacts below.
        df = self._validate_rows(pd.read_csv(raw_dataset_path))
        df.to_csv(validated_data_path)
        
        # Preserve sentiment-class balance while keeping the split reproducible.
        trainset, testset = train_test_split(
            df,
            test_size=0.2,
            random_state=cfg['seed'],
            stratify=df[self.LABEL_COLUMN],
        )
        
        trainset.to_csv(train_dataset_path, index=False)
        testset.to_csv(test_dataset_path, index=False)

        if valid_dataset_path.exists():
            valid_dataset = pd.read_csv(valid_dataset_path)
            valid_dataset.to_csv(valid_dataset_path, index=False)

        return {
            "validated_data": validated_data_path,
            "train_dataset": train_dataset_path,
            "test_dataset": test_dataset_path,
            "validation_dataset": valid_dataset_path,
        }

    def _validate_rows(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Perform deterministic, safe pre-split row validation."""
        required_columns = {self.TEXT_COLUMN, self.LABEL_COLUMN}
        missing_columns = required_columns - set(dataframe.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

        df = dataframe.copy()
        initial_count = len(df)

        # Do not impute text or labels: synthetic values would manufacture
        # training features/targets. Drop unusable records instead.
        df = df.dropna(subset=[self.TEXT_COLUMN, self.LABEL_COLUMN])
        df[self.TEXT_COLUMN] = df[self.TEXT_COLUMN].astype(str).str.strip()
        df = df[df[self.TEXT_COLUMN].ne("")]

        # The downstream sentiment model expects -1, 0, and 1.
        df[self.LABEL_COLUMN] = pd.to_numeric(df[self.LABEL_COLUMN], errors="coerce")
        df = df[df[self.LABEL_COLUMN].isin(self.VALID_LABELS)]
        df[self.LABEL_COLUMN] = df[self.LABEL_COLUMN].astype(int)

        # Remove repeated text-label pairs. Same text with conflicting labels is
        # retained intentionally so it can be reviewed as a data-quality issue.
        df = df.drop_duplicates(subset=[self.TEXT_COLUMN, self.LABEL_COLUMN])
        df = df.reset_index(drop=True)

        if df.empty:
            raise ValueError("No valid rows remain after data validation.")
        if df[self.LABEL_COLUMN].nunique() < 2:
            raise ValueError("At least two label classes are required to split the data.")
        if df[self.LABEL_COLUMN].value_counts().min() < 2:
            raise ValueError("Each label class needs at least two rows for a stratified split.")

        print(f"Validated {len(df):,} of {initial_count:,} rows; raw CSV was not modified.")
        return df
