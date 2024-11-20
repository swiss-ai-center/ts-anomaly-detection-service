import asyncio
import time
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from common_code.config import get_settings
from common_code.http_client import HttpClient
from common_code.logger.logger import get_logger, Logger
from common_code.service.controller import router as service_router
from common_code.service.service import ServiceService
from common_code.storage.service import StorageService
from common_code.tasks.controller import router as tasks_router
from common_code.tasks.service import TasksService
from common_code.tasks.models import TaskData
from common_code.service.models import Service
from common_code.service.enums import ServiceStatus
from common_code.common.enums import FieldDescriptionType, ExecutionUnitTagName, ExecutionUnitTagAcronym
from common_code.common.models import FieldDescription, ExecutionUnitTag
from contextlib import asynccontextmanager

# Imports required by the service's model
from momentfm import MOMENTPipeline
# from momentfm.utils.anomaly_detection_metrics import adjbestf1

import torch
from torch.utils.data import DataLoader

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from tqdm import tqdm
# from pprint import pprint
import io

settings = get_settings()


class AnomalyDetectionDataset:
    def __init__(
        self,
        data_split: str = "train",
        data_stride_len: int = 512,
        random_seed: int = 42,
        data: str = "data/198_UCR_Anomaly_tiltAPB2_50000_124159_124985.out"

    ):
        """
        Parameters
        ----------
        data_split : str
            Split of the dataset, 'train', or 'test'
        data_stride_len : int
            Stride length for the data.
        random_seed : int
            Random seed for reproducibility.
        file_path : str
            Path to the data file.
        """

        self.data = data
        self.series = "198_UCR_Anomaly_tiltAPB2_50000_124159_124985"
        self.data_split = data_split
        self.data_stride_len = data_stride_len
        self.random_seed = random_seed
        self.seq_len = 512

        # Downsampling for experiments. Refer
        # https://github.com/mononitogoswami/tsad-model-selection for more details
        self.downsampling_factor = 10
        self.min_length = (
            2560  # Minimum length of time-series after downsampling for experiments
        )

        # Read data
        self._read_data()

    def _get_borders(self):
        details = self.series.split("_")
        n_train = int(details[4])
        train_end = n_train
        test_start = train_end

        return slice(0, train_end), slice(test_start, None)

    def _read_data(self):
        self.scaler = StandardScaler()

        raw = self.data["dataset"].data
        input_type = self.data["dataset"].type

        print("Input type: ", str(input_type))

        df = pd.read_csv(io.BytesIO(raw), header=None)

        df.interpolate(inplace=True, method="cubic")

        self.length_timeseries = len(df)
        self.n_channels = 1
        labels = df.iloc[:, -1].values
        timeseries = df.iloc[:, 0].values.reshape(-1, 1)

        data_splits = self._get_borders()

        self.scaler.fit(timeseries[data_splits[0]])
        timeseries = self.scaler.transform(timeseries)
        timeseries = timeseries.squeeze()

        if self.data_split == "train":
            self.data, self.labels = timeseries[data_splits[0]], labels[data_splits[0]]
        elif self.data_split == "test":
            self.data, self.labels = timeseries[data_splits[1]], labels[data_splits[1]]

        self.length_timeseries = self.data.shape[0]

    def __getitem__(self, index):
        seq_start = self.data_stride_len * index
        seq_end = seq_start + self.seq_len
        input_mask = np.ones(self.seq_len)

        if seq_end > self.length_timeseries:
            seq_start = self.length_timeseries - self.seq_len
            seq_end = None

        timeseries = self.data[seq_start:seq_end].reshape(
            (self.n_channels, self.seq_len)
        )
        labels = (
            self.labels[seq_start:seq_end]
            .astype(int)
            .reshape((self.n_channels, self.seq_len))
        )

        return timeseries, input_mask, labels

    def __len__(self):
        return (self.length_timeseries // self.data_stride_len) + 1


class MyService(Service):
    """
    This service takes as input a time series.
    The service returns a plot of the time series with the detected anomalies.
    """

    # Any additional fields must be excluded for Pydantic to work
    _model: object
    _logger: Logger

    def __init__(self):
        super().__init__(
            name="Time Series Anomaly Detection",
            slug="ts-anomaly-detection",
            url=settings.service_url,
            summary=api_summary,
            description=api_description,
            status=ServiceStatus.AVAILABLE,
            data_in_fields=[
                FieldDescription(
                    name="dataset",
                    type=[
                        FieldDescriptionType.TEXT_CSV,
                    ],
                ),
            ],
            data_out_fields=[
                FieldDescription(
                    name="result", type=[FieldDescriptionType.IMAGE_PNG]
                ),
            ],
            tags=[
                ExecutionUnitTag(
                    name=ExecutionUnitTagName.ANOMALY_DETECTION,
                    acronym=ExecutionUnitTagAcronym.ANOMALY_DETECTION,
                ),
            ],
            has_ai=True,
            docs_url="https://docs.swiss-ai-center.ch/reference/services/ts-anomaly-detection/",
        )
        self._logger = get_logger(settings)

    def process(self, data):
        # NOTE that the data is a dictionary with the keys being the field names set in the data_in_fields
        # The objects in the data variable are always bytes. It is necessary to convert them to the desired type
        # before using them.
        # raw = data["image"].data
        # input_type = data["image"].type
        # ... do something with the raw data

        model = MOMENTPipeline.from_pretrained(
            "AutonLab/MOMENT-1-large",
            model_kwargs={"task_name": "reconstruction"},
            # For anomaly detection, we will load MOMENT in `reconstruction` mode
        )

        model.init()

        # raw = data["input_time_series"].data
        test_dataset = AnomalyDetectionDataset(data_split='test', random_seed=13,
                                               data=data)
        # Anomaly Detection using MOMENT
        # Now we will use MOMENT to detect anomalies  scores for the time series.

        test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False, drop_last=False)

        model = model.float()

        trues, preds, labels = [], [], []
        with torch.no_grad():
            for batch_x, batch_masks, batch_labels in tqdm(test_dataloader, total=len(test_dataloader)):
                batch_x = batch_x.float()
                batch_masks = batch_masks

                output = model(batch_x, input_mask=batch_masks)  # [batch_size, n_channels, window_size]

                trues.append(batch_x.detach().squeeze().cpu().numpy())
                preds.append(output.reconstruction.detach().squeeze().cpu().numpy())
                labels.append(batch_labels.detach().cpu().numpy())

        trues = np.concatenate(trues, axis=0).flatten()
        preds = np.concatenate(preds, axis=0).flatten()
        labels = np.concatenate(labels, axis=0).flatten()

        # The last and the second to last windows have overlapping timesteps
        # We will remove these overlapping predictions
        n_unique_timesteps = 512 - trues.shape[0] + test_dataset.length_timeseries
        trues = np.concatenate([trues[:512 * (test_dataset.length_timeseries // 512)], trues[-n_unique_timesteps:]])
        preds = np.concatenate([preds[:512 * (test_dataset.length_timeseries // 512)], preds[-n_unique_timesteps:]])
        labels = np.concatenate([labels[:512 * (test_dataset.length_timeseries // 512)], labels[-n_unique_timesteps:]])
        assert trues.shape[0] == test_dataset.length_timeseries

        # We will use the Mean Squared Error (MSE) between the observed values and
        # MOMENT's predictions as the anomaly score
        anomaly_scores = (trues - preds) ** 2

        # result = f"Zero-shot Adjusted Best F1 Score: {adjbestf1(y_true=labels, y_scores=anomaly_scores)}"

        anomaly_start = 74158
        anomaly_end = 74984
        start = anomaly_start - 512
        end = anomaly_end + 512

        plt.plot(trues[start:end], label="Observed", c='darkblue')
        plt.plot(preds[start:end], label="Predicted", c='red')
        plt.plot(anomaly_scores[start:end], label="Anomaly Score", c='black')
        plt.legend(fontsize=16)

        # Save the plot
        buf = io.BytesIO()
        plt.savefig(buf, format='png')

        # Reset the buffer
        buf.seek(0)

        # NOTE that the result must be a dictionary with the keys being the field names set in the data_out_fields
        return {
            "result": TaskData(data=buf.read(), type=FieldDescriptionType.IMAGE_PNG)
        }


service_service: ServiceService | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Manual instances because startup events doesn't support Dependency Injection
    # https://github.com/tiangolo/fastapi/issues/2057
    # https://github.com/tiangolo/fastapi/issues/425

    # Global variable
    global service_service

    # Startup
    logger = get_logger(settings)
    http_client = HttpClient()
    storage_service = StorageService(logger)
    my_service = MyService()
    tasks_service = TasksService(logger, settings, http_client, storage_service)
    service_service = ServiceService(logger, settings, http_client, tasks_service)

    tasks_service.set_service(my_service)

    # Start the tasks service
    tasks_service.start()

    async def announce():
        retries = settings.engine_announce_retries
        for engine_url in settings.engine_urls:
            announced = False
            while not announced and retries > 0:
                announced = await service_service.announce_service(my_service, engine_url)
                retries -= 1
                if not announced:
                    time.sleep(settings.engine_announce_retry_delay)
                    if retries == 0:
                        logger.warning(
                            f"Aborting service announcement after "
                            f"{settings.engine_announce_retries} retries"
                        )

    # Announce the service to its engine
    asyncio.ensure_future(announce())

    yield

    # Shutdown
    for engine_url in settings.engine_urls:
        await service_service.graceful_shutdown(my_service, engine_url)


api_description = """This service takes as input a time series (.csv file) and returns the anomaly score
(adjusted best F1) in a zero-shot fashion.
"""
api_summary = """This service detects anomalies in univariate time series using the MOMENT library.

"""

# Define the FastAPI application with information
app = FastAPI(
    lifespan=lifespan,
    title="Time Series Anomaly Detection API.",
    description=api_description,
    version="0.0.1",
    contact={
        "name": "Swiss AI Center",
        "url": "https://swiss-ai-center.ch/",
        "email": "info@swiss-ai-center.ch",
    },
    swagger_ui_parameters={
        "tagsSorter": "alpha",
        "operationsSorter": "method",
    },
    license_info={
        "name": "MIT License",
        "url": "https://choosealicense.com/licenses/mit/",
    },
)

# Include routers from other files
app.include_router(service_router, tags=["Service"])
app.include_router(tasks_router, tags=["Tasks"])

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Redirect to docs
@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse("/docs", status_code=301)
