import argparse
import numpy as np
import tqdm
from onnxruntime.quantization import quantize_static, quantize_dynamic, QuantType, CalibrationMethod
from onnxruntime.quantization import CalibrationDataReader
from fddbenchmark import FDDDataset
from sklearn.preprocessing import StandardScaler
import os
import tempfile
import subprocess

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_onnx', type=str, required=True)
    parser.add_argument('--output_onnx', type=str, default='model_quantized.onnx')
    parser.add_argument('--dataset', type=str, default='reinartz_tep')
    parser.add_argument('--window_size', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--calibration_samples', type=int, default=320)
    parser.add_argument('--quant_type', type=str, default='static', choices=['static', 'dynamic'])
    parser.add_argument('--quant_mode', type=str, default='QInt8', choices=['QInt8', 'QUInt8'])
    parser.add_argument('--preprocess', action='store_true')
    return parser.parse_args()

class BatchDataReader(CalibrationDataReader):
    def __init__(self, samples, batch_size):
        self.batch_size = batch_size
        self.batches = []
        for i in range(0, len(samples), batch_size):
            batch = np.stack(samples[i:i+batch_size], axis=0)
            self.batches.append(batch)
        self.idx = 0
    def get_next(self):
        if self.idx >= len(self.batches):
            return None
        batch = self.batches[self.idx]
        self.idx += 1
        return {'batch_ts': batch}

def get_calibration_samples_fast(args):
    dataset = FDDDataset(name=args.dataset)
    train_df = dataset.df[dataset.train_mask]
    max_start = len(train_df) - args.window_size
    if max_start <= 0:
        raise ValueError("Not enough data for one window")
    n_windows = min(args.calibration_samples, max_start)
    start_indices = np.random.choice(max_start, n_windows, replace=False)
    scaler = StandardScaler()
    scaler.fit(train_df)
    samples = []
    for start in tqdm.tqdm(start_indices, desc="Extracting windows"):
        window = train_df.iloc[start:start+args.window_size].values
        window_scaled = scaler.transform(window)
        window_scaled = window_scaled.T.astype(np.float32)
        samples.append(window_scaled)
    return samples

def preprocess_onnx(input_path, output_path):
    print("Running ONNX preprocessing...")
    subprocess.run([
        "python", "-m", "onnxruntime.quantization.preprocess",
        "--input", input_path,
        "--output", output_path
    ], check=True)
    return output_path

def main():
    args = parse_args()
    model_to_quant = args.input_onnx
    if args.preprocess:
        preprocessed = tempfile.NamedTemporaryFile(suffix=".onnx", delete=False).name
        preprocess_onnx(args.input_onnx, preprocessed)
        model_to_quant = preprocessed

    if args.quant_type == 'static':
        print("Collecting calibration samples...")
        raw_samples = get_calibration_samples_fast(args)
        print(f"Collected {len(raw_samples)} raw samples")
        data_reader = BatchDataReader(raw_samples, args.batch_size)
        print(f"Created {len(data_reader.batches)} batches")
        activation_type = QuantType.QInt8 if args.quant_mode == 'QInt8' else QuantType.QUInt8
        weight_type = QuantType.QInt8 if args.quant_mode == 'QInt8' else QuantType.QUInt8
        print("Starting static quantization...")
        quantize_static(
            model_input=model_to_quant,
            model_output=args.output_onnx,
            calibration_data_reader=data_reader,
            quant_format=QuantType.QInt8,
            per_channel=True,
            activation_type=activation_type,
            weight_type=weight_type,
            calibrate_method=CalibrationMethod.MinMax,
            use_external_data_format=False,
            reduce_range=False
        )
    else:
        weight_type = QuantType.QInt8 if args.quant_mode == 'QInt8' else QuantType.QUInt8
        print("Starting dynamic quantization...")
        quantize_dynamic(
            model_input=model_to_quant,
            model_output=args.output_onnx,
            weight_type=weight_type
        )
    print(f"Quantization finished. Saved to {args.output_onnx}")
    if args.preprocess and os.path.exists(preprocessed):
        os.unlink(preprocessed)

if __name__ == '__main__':
    main()