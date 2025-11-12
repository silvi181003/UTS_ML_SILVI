#nullable disable
using System;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Vision;

namespace deteksi_sampah
{
    public class ImageData
    {
        [LoadColumn(0)]
        public string ImagePath { get; set; }

        [LoadColumn(1)]
        public string Label { get; set; }
    }

    class Program
    {
        static readonly string _assetsPath = Path.Combine(Environment.CurrentDirectory, "assets");
        static readonly string _imagesFolder = Path.Combine(_assetsPath, "images");
        static readonly string _trainTagsTsv = Path.Combine(_imagesFolder, "tags.tsv");
        static readonly string _testTagsTsv = Path.Combine(_imagesFolder, "test-tags.tsv");
        static readonly string _modelPath = Path.Combine(_assetsPath, "ModelDeteksiSampah.zip");

        static void Main(string[] args)
        {
            Console.WriteLine("╔════════════════════════════════════════════════════════════╗");
            Console.WriteLine("║     SISTEM DETEKSI SAMPAH - DINAS PERSAMPAHAN KONOHA      ║");
            Console.WriteLine("╚════════════════════════════════════════════════════════════╝\n");

            if (args.Length > 0 && args[0] == "setup")
            {
                SetupDataset.Setup();
                return;
            }

            if (args.Length > 0 && args[0] == "predict")
            {
                if (args.Length < 2)
                {
                    Console.WriteLine("Usage: dotnet run predict <path-to-image>");
                    return;
                }
                TestModel.PredictImage(args[1]);
                return;
            }

            if (!File.Exists(_trainTagsTsv))
            {
                Console.ForegroundColor = ConsoleColor.Yellow;
                Console.WriteLine("⚠ Dataset belum di-setup!\nJalankan: dotnet run setup");
                Console.ResetColor();
                return;
            }

            MLContext mlContext = new MLContext(seed: 1);

            try
            {
                Console.WriteLine("[1] Loading dataset...");
                IDataView trainingData = LoadData(mlContext, _trainTagsTsv);
                IDataView testData = LoadData(mlContext, _testTagsTsv);

                Console.WriteLine("\n[2] Training model...");
                var model = BuildAndTrainModel(mlContext, trainingData);

                Console.WriteLine("\n[3] Evaluasi model...");
                EvaluateModel(mlContext, testData, model);

                Console.WriteLine("\n[4] Menyimpan model...");
                SaveModel(mlContext, model, trainingData.Schema);

                Console.WriteLine("\n✓ TRAINING SELESAI!");
                Console.WriteLine($"Model tersimpan di: {_modelPath}");
            }
            catch (Exception ex)
            {
                Console.ForegroundColor = ConsoleColor.Red;
                Console.WriteLine($"\n❌ ERROR: {ex.Message}");
                Console.ResetColor();
            }
        }

        static IDataView LoadData(MLContext mlContext, string dataPath)
        {
            var data = mlContext.Data.LoadFromTextFile<ImageData>(
                path: dataPath, hasHeader: true, separatorChar: '\t');
            var count = data.GetColumn<string>("ImagePath").Count();
            Console.WriteLine($"    ✓ Loaded {count} gambar");
            return data;
        }

        static ITransformer BuildAndTrainModel(MLContext mlContext, IDataView trainingData)
        {
            var pipeline = mlContext.Transforms.Conversion
                .MapValueToKey(inputColumnName: "Label", outputColumnName: "LabelAsKey")
                .Append(mlContext.Transforms.LoadRawImageBytes(
                    outputColumnName: "Image",
                    imageFolder: _imagesFolder,
                    inputColumnName: "ImagePath"))
                .Append(mlContext.MulticlassClassification.Trainers.ImageClassification(
                    new ImageClassificationTrainer.Options()
                    {
                        FeatureColumnName = "Image",
                        LabelColumnName = "LabelAsKey",
                        Arch = ImageClassificationTrainer.Architecture.ResnetV2101,
                        Epoch = 10,
                        TestOnTrainSet = true,
                        EarlyStoppingCriteria = null,
                    }))
                .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            Console.WriteLine("    Training dengan ResNet V2-101...");
            var model = pipeline.Fit(trainingData);
            return model;
        }

        static void EvaluateModel(MLContext mlContext, IDataView testData, ITransformer model)
        {
            var predictions = model.Transform(testData);
            var metrics = mlContext.MulticlassClassification.Evaluate(
                predictions, labelColumnName: "LabelAsKey", predictedLabelColumnName: "PredictedLabel");

            Console.WriteLine($"    Macro Accuracy: {metrics.MacroAccuracy:P2}");
            Console.WriteLine($"    Micro Accuracy: {metrics.MicroAccuracy:P2}");
            Console.WriteLine($"    Log Loss: {metrics.LogLoss:F4}");
        }

        static void SaveModel(MLContext mlContext, ITransformer model, DataViewSchema schema)
        {
            mlContext.Model.Save(model, schema, _modelPath);
            Console.WriteLine($"    ✓ Model saved: {Path.GetFileName(_modelPath)}");
        }
    }
}