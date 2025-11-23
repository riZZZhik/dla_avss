from pathlib import Path

import pandas as pd

from src.logger.utils import plot_spectrogram, plot_waveform
from src.metrics.tracker import MetricTracker
from src.metrics.utils import calc_pesq, calc_sdr, calc_si_snr, calc_stoi
from src.trainer.base_trainer import BaseTrainer


class Trainer(BaseTrainer):
    """
    Trainer class. Defines the logic of batch logging and processing.
    """

    def process_batch(self, batch, metrics: MetricTracker):
        """
        Run batch through the model, compute metrics, compute loss,
        and do training step (during training stage).

        The function expects that criterion aggregates all losses
        (if there are many) into a single one defined in the 'loss' key.

        Args:
            batch (dict): dict-based batch containing the data from
                the dataloader.
            metrics (MetricTracker): MetricTracker object that computes
                and aggregates the metrics. The metrics depend on the type of
                the partition (train or inference).
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform),
                model outputs, and losses.
        """
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)  # transform batch on device -- faster

        metric_funcs = self.metrics["inference"]
        if self.is_train:
            metric_funcs = self.metrics["train"]
            self.optimizer.zero_grad()

        outputs = self.model(**batch)
        batch.update(outputs)

        all_losses = self.criterion(**batch)
        batch.update(all_losses)

        if self.is_train:
            batch["loss"].backward()  # sum of all losses is always called loss
            self._clip_grad_norm()
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        # update metrics for each loss (in case of multiple losses)
        for loss_name in self.config.writer.loss_names:
            metrics.update(loss_name, batch[loss_name].item())

        for met in metric_funcs:
            metrics.update(met.name, met(**batch))
        return batch

    def _log_batch(self, batch_idx, batch, mode="train"):
        """
        Log data from batch. Calls self.writer.add_* to log data
        to the experiment tracker.

        Args:
            batch_idx (int): index of the current batch.
            batch (dict): dict-based batch after going through
                the 'process_batch' function.
            mode (str): train or inference. Defines which logging
                rules to apply.
        """
        # method to log data from you batch
        # such as audio, text or images, for example

        # logging scheme might be different for different partitions
        if mode == "train":  # the method is called only every self.log_step steps
            self.log_waveform(**batch)
        else:
            # Log Stuff
            self.log_waveform(**batch)
            self.log_predictions(**batch)

    def log_spectrogram(self, spectrogram, **batch):
        spectrogram_for_plot = spectrogram[0].detach().cpu()
        image = plot_spectrogram(spectrogram_for_plot)
        self.writer.add_image("spectrogram", image)

    def log_waveform(self, preds, **batch):
        waveform_s1_for_plot = preds[0][0].detach().cpu()
        waveform_s2_for_plot = preds[0][1].detach().cpu()
        image_s1 = plot_waveform(waveform_s1_for_plot)
        image_s2 = plot_waveform(waveform_s2_for_plot)
        self.writer.add_image("waveform", image_s1)
        self.writer.add_image("waveform", image_s2)

    def log_predictions(self, preds, speakers, audio_path, examples_to_log=4, **batch):
        tuples = list(zip(preds, speakers, audio_path))

        for preds, speakers, audio_path in tuples[:examples_to_log]:
            predicted_s1 = preds[..., 0, :]
            speaker1 = speakers[..., 0, :]
            predicted_s2 = preds[..., 1, :]
            speaker2 = speakers[..., 1, :]

            metadata_s1 = {"si_snri": calc_si_snr(predicted_s1, speaker1)}
            metadata_s2 = {"si_snri": calc_si_snr(predicted_s2, speaker2)}

            self.writer.add_audio(
                audio_name=audio_path.replace(".", "_original_s1."), audio=speaker1
            )
            self.writer.add_audio(
                audio_name=audio_path.replace(".", "_original_s2."), audio=speaker2
            )
            self.writer.add_audio(
                audio_name=audio_path.replace(".", "_predicted_s1."),
                audio=predicted_s1,
                metadata=metadata_s1,
            )
            self.writer.add_audio(
                audio_name=audio_path.replace(".", "_predicted_s2."),
                audio=predicted_s2,
                metadata=metadata_s2,
            )
