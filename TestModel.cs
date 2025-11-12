#nullable disable
using System;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace deteksi_sampah
{
    public class ModelInput
    {
        public byte[] Image { get; set; }
        public uint LabelAsKey { get; set; }
        public string ImagePath { get; set; }
        public string Label { get; set; }
    }

    public class ModelOutput
    {
        public string ImagePath { get; set; }
        public string Label { get; set; }
        public string PredictedLabel { get; set; }
    }

    public class TestModel
    {
        static readonly string _modelPath = "assets/ModelDeteksiSampah.zip";

        public static void PredictImage(string imagePath)
        {
            if (!File.Exists(_modelPath))
            {
                Console.ForegroundColor = ConsoleColor.Red;
                Console.WriteLine("❌ Model belum ada! Jalankan training dulu dengan: dotnet run");
                Console.ResetColor();
                return;
            }

            string fixedPath = imagePath;
            if (!File.Exists(fixedPath))
            {
                var possiblePath = Path.Combine("assets", "images", imagePath.Replace('/', Path.DirectorySeparatorChar));
                if (File.Exists(possiblePath))
                    fixedPath = possiblePath;
                else
                {
                    Console.ForegroundColor = ConsoleColor.Red;
                    Console.WriteLine($"❌ Gambar tidak ditemukan: {imagePath}");
                    Console.ResetColor();
                    return;
                }
            }

            try
            {
                MLContext mlContext = new MLContext();
                
                // Load model
                ITransformer model = mlContext.Model.Load(_modelPath, out var schema);
                
                // Create prediction engine
                var predEngine = mlContext.Model.CreatePredictionEngine<ModelInput, ModelOutput>(model);
                
                // Predict
                string relativePath = fixedPath;
                if (relativePath.Contains("assets/images/"))
                    relativePath = relativePath.Substring(relativePath.IndexOf("assets/images/") + "assets/images/".Length);

                var input = new ModelInput { ImagePath = relativePath };
                var prediction = predEngine.Predict(input);

                Console.WriteLine("\n╔════════════════════════════════════════╗");
                Console.WriteLine("║         HASIL PREDIKSI                 ║");
                Console.WriteLine("╚════════════════════════════════════════╝");
                Console.WriteLine($"  Gambar   : {Path.GetFileName(imagePath)}");
                Console.WriteLine($"  Path     : {imagePath}");
                Console.WriteLine($"  Kategori : {prediction.PredictedLabel}");
                Console.WriteLine("════════════════════════════════════════════\n");

                // Rekomendasi pembuangan
                ShowDisposalRecommendation(prediction.PredictedLabel);
            }
            catch (Exception ex)
            {
                Console.ForegroundColor = ConsoleColor.Red;
                Console.WriteLine($"❌ Error saat prediksi: {ex.Message}");
                Console.ResetColor();
            }
        }

        static void ShowDisposalRecommendation(string category)
        {
            Console.WriteLine("📌 REKOMENDASI PEMBUANGAN:");
            
            switch (category?.ToLower())
            {
                case "plastik":
                    Console.WriteLine("   • Tempat sampah: KUNING");
                    Console.WriteLine("   • Dapat didaur ulang");
                    Console.WriteLine("   • Bersihkan sebelum dibuang");
                    break;
                case "kertas":
                    Console.WriteLine("   • Tempat sampah: BIRU");
                    Console.WriteLine("   • Dapat didaur ulang");
                    Console.WriteLine("   • Lipat/ratakan untuk menghemat ruang");
                    break;
                case "logam":
                    Console.WriteLine("   • Tempat sampah: BIRU");
                    Console.WriteLine("   • Dapat didaur ulang");
                    Console.WriteLine("   • Pisahkan dari sampah lain");
                    break;
                case "organik":
                    Console.WriteLine("   • Tempat sampah: HIJAU");
                    Console.WriteLine("   • Dapat dijadikan kompos");
                    Console.WriteLine("   • Proses penguraian alami");
                    break;
                case "kaca":
                    Console.WriteLine("   • Tempat sampah: BIRU");
                    Console.WriteLine("   • Dapat didaur ulang");
                    Console.WriteLine("   • Hati-hati pecahan tajam");
                    break;
                default:
                    Console.WriteLine("   • Kategori tidak dikenali");
                    break;
            }
            Console.WriteLine();
        }
    }
}