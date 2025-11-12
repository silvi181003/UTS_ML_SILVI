
#nullable disable
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;

namespace deteksi_sampah
{
    public class SetupDataset
    {
        static readonly string _baseFolder = Path.Combine(Environment.CurrentDirectory, "assets");
        static readonly string _imagesFolder = Path.Combine(_baseFolder, "images");
        static readonly string _trainFolder = Path.Combine(_imagesFolder, "train");
        static readonly string _testFolder = Path.Combine(_imagesFolder, "test");
        static readonly string[] _categories = { "plastik", "kertas", "logam", "organik", "kaca" };

        public static void Setup()
        {
            Console.WriteLine("╔════════════════════════════════════════════════════════════╗");
            Console.WriteLine("║  SETUP DATASET DETEKSI SAMPAH - DINAS PERSAMPAHAN KONOHA  ║");
            Console.WriteLine("╚════════════════════════════════════════════════════════════╝\n");

            CreateFolderStructure();
            
            var trainImages = ScanImagesInFolder(_trainFolder);
            var testImages = ScanImagesInFolder(_testFolder);

            Console.WriteLine($"\n✓ Ditemukan {CountImages(trainImages)} gambar training");
            Console.WriteLine($"✓ Ditemukan {CountImages(testImages)} gambar test");

            GenerateTagsFile(trainImages, Path.Combine(_imagesFolder, "tags.tsv"));
            GenerateTagsFile(testImages, Path.Combine(_imagesFolder, "test-tags.tsv"));

            ShowSummary(trainImages, testImages);
            CreateDownloadGuide();

            Console.WriteLine("\n✓ Setup dataset selesai!");
        }

        static void CreateFolderStructure()
        {
            Console.WriteLine("[1] Membuat struktur folder...\n");

            Directory.CreateDirectory(_baseFolder);
            Directory.CreateDirectory(_imagesFolder);
            Directory.CreateDirectory(_trainFolder);
            Directory.CreateDirectory(_testFolder);

            foreach (var category in _categories)
            {
                Directory.CreateDirectory(Path.Combine(_trainFolder, category));
                Directory.CreateDirectory(Path.Combine(_testFolder, category));
            }

            Directory.CreateDirectory(Path.Combine(_baseFolder, "inception"));

            Console.WriteLine("   ✓ Struktur folder berhasil dibuat!");
        }

        static Dictionary<string, List<string>> ScanImagesInFolder(string baseFolder)
        {
            var imagesByCategory = new Dictionary<string, List<string>>();
            var extensions = new[] { ".jpg", ".jpeg", ".png", ".bmp" };

            foreach (var category in _categories)
            {
                var categoryPath = Path.Combine(baseFolder, category);
                var images = Directory.Exists(categoryPath) 
                    ? Directory.GetFiles(categoryPath).Where(f => extensions.Contains(Path.GetExtension(f).ToLower())).ToList()
                    : new List<string>();
                imagesByCategory[category] = images;
            }

            return imagesByCategory;
        }

        static void GenerateTagsFile(Dictionary<string, List<string>> imagesByCategory, string outputPath)
        {
            var sb = new StringBuilder();
            sb.AppendLine("ImagePath\tLabel");

            foreach (var category in _categories)
            {
                if (imagesByCategory.ContainsKey(category))
                {
                    foreach (var imagePath in imagesByCategory[category])
                    {
                        var relativePath = imagePath.Replace(_imagesFolder + Path.DirectorySeparatorChar, "")
                                                    .Replace(Path.DirectorySeparatorChar, '/');
                        var label = char.ToUpper(category[0]) + category.Substring(1);
                        sb.AppendLine($"{relativePath}\t{label}");
                    }
                }
            }

            File.WriteAllText(outputPath, sb.ToString(), Encoding.UTF8);
            Console.WriteLine($"✓ {Path.GetFileName(outputPath)} berhasil dibuat");
        }

        static void ShowSummary(Dictionary<string, List<string>> trainImages, Dictionary<string, List<string>> testImages)
        {
            Console.WriteLine("\n[Ringkasan Dataset]");
            Console.WriteLine("┌──────────────┬──────────┬──────────┐");
            Console.WriteLine("│ Kategori     │ Training │   Test   │");
            Console.WriteLine("├──────────────┼──────────┼──────────┤");

            foreach (var category in _categories)
            {
                int trainCount = trainImages[category].Count;
                int testCount = testImages[category].Count;
                var name = char.ToUpper(category[0]) + category.Substring(1);
                Console.WriteLine($"│ {name,-12} │   {trainCount,4}   │   {testCount,4}   │");
            }
            Console.WriteLine("└──────────────┴──────────┴──────────┘");
        }

        static void CreateDownloadGuide()
        {
            var guidePath = Path.Combine(_baseFolder, "CARA_DOWNLOAD_DATASET.txt");
            var guide = @"═══════════════════════════════════════════════════════════════
PANDUAN DOWNLOAD DATASET SAMPAH
═══════════════════════════════════════════════════════════════

SUMBER DATASET RECOMMENDED:

1. Kaggle - Waste Classification Data
   URL: https://www.kaggle.com/datasets/techsash/waste-classification-data
   
2. GitHub - TrashNet
   URL: https://github.com/garythung/trashnet
   
3. Roboflow Universe
   URL: https://universe.roboflow.com/search?q=trash

CARA DOWNLOAD:
1. Pilih salah satu sumber di atas
2. Download dataset (biasanya format ZIP)
3. Extract file
4. Pindahkan gambar ke folder yang sesuai:
   
   assets/images/train/plastik/   ← Gambar sampah plastik
   assets/images/train/kertas/    ← Gambar sampah kertas
   assets/images/train/logam/     ← Gambar sampah logam
   assets/images/train/organik/   ← Gambar sampah organik
   assets/images/train/kaca/      ← Gambar sampah kaca

5. Lakukan hal yang sama untuk folder test/
6. Jalankan setup lagi: dotnet run setup

REKOMENDASI JUMLAH:
- Training: Minimal 100 gambar per kategori
- Test: Minimal 20 gambar per kategori

═══════════════════════════════════════════════════════════════";
            
            File.WriteAllText(guidePath, guide);
            Console.WriteLine($"\n✓ Panduan tersimpan di: CARA_DOWNLOAD_DATASET.txt");
        }

        static int CountImages(Dictionary<string, List<string>> images)
        {
            return images.Values.Sum(list => list.Count);
        }
    }
}