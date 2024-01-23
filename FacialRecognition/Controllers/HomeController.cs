using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;
using Emgu.CV;
using Emgu.CV.Structure;
using Emgu.CV.Face;
using System.Threading.Tasks;
using Emgu.CV.Util;
using Emgu.CV.Reg;
using Emgu.CV.CvEnum;
using System;
using System.Collections.Generic;
using System.Drawing;

namespace FacialRecognition.Controllers
{
	[Route("api/[controller]")]
	[ApiController]
	public class HomeController : ControllerBase
	{


		[HttpPut("MatchFaces")]
		public async Task<IActionResult> MatchFaces(IFormFile FirstSample, IFormFile SecondSample, IFormFile PredictSample)
		{

			string HaarCascadePath = Directory.GetCurrentDirectory() + @"\haarcascade_frontalface_alt.xml";
			CascadeClassifier faceCasacdeClassifier = new CascadeClassifier(HaarCascadePath);
			EigenFaceRecognizer recognizer = new EigenFaceRecognizer();
			VectorOfMat TrainImageVecofMat = new VectorOfMat();
			List<int> Labels = new List<int>();
			VectorOfInt vectorOfInt = new VectorOfInt();
			Labels.Add(0);
			Labels.Add(1);

			var FirstStream = FirstSample.OpenReadStream();
			var SecondStream = SecondSample.OpenReadStream();
			var MemoryFirstStream = new MemoryStream();
			var MemorySecondStream = new MemoryStream();
			FirstStream.CopyTo(MemoryFirstStream);
			SecondStream.CopyTo(MemorySecondStream);
			byte[] FirstTrainBytes = MemoryFirstStream.ToArray();
			byte[] SecondTrainBytes = MemorySecondStream.ToArray();

			Mat FirstTrainMat = new Mat();
			Mat SecondTrainMat = new Mat();


			CvInvoke.Imdecode(FirstTrainBytes, ImreadModes.Color, FirstTrainMat);
			CvInvoke.Imdecode(SecondTrainBytes, ImreadModes.Color, SecondTrainMat);
			Rectangle[] FirstImageFace = faceCasacdeClassifier.DetectMultiScale(FirstTrainMat, 1.1, 3, Size.Empty, Size.Empty);
			Image<Gray, Byte> FitstTrainEmguImage = FirstTrainMat.ToImage<Gray, Byte>();
			Image<Gray, Byte> SecondTrainEmguImage = SecondTrainMat.ToImage<Gray, Byte>().Resize(500, 500, Inter.Cubic).Convert<Gray, Byte>();
			FitstTrainEmguImage.ROI = FirstImageFace[0];
			Image<Gray, Byte> ROIFirstImage = FitstTrainEmguImage.Resize(500, 500, Inter.Cubic).Convert<Gray, Byte>();
			ROIFirstImage.Save("C:\\Users\\motaz\\Desktop\\train.jpg");
			FitstTrainEmguImage.Resize(500, 500, Inter.Cubic).Convert<Gray, Byte>();
			List<Image<Gray, Byte>> TrainedFaces = new List<Image<Gray, Byte>>();
			TrainedFaces.Add(ROIFirstImage);
			TrainedFaces.Add(SecondTrainEmguImage);
			TrainImageVecofMat.Push(TrainedFaces.ToArray());
			vectorOfInt.Push(Labels.ToArray());

			recognizer.Train(TrainImageVecofMat, vectorOfInt);

			 var PredictStream = PredictSample.OpenReadStream();
			 var MemoryPredictStream = new MemoryStream();
			PredictStream.CopyTo(MemoryPredictStream);
			byte[] PredictBytes = MemoryPredictStream.ToArray();
			Mat PredictMat = new Mat();
			CvInvoke.Imdecode(PredictBytes, ImreadModes.Color, PredictMat);
			Rectangle[] PredictImageFace = faceCasacdeClassifier.DetectMultiScale(PredictMat, 1.1, 3, Size.Empty, Size.Empty);

			Image<Gray, byte> PredictEmguImage = PredictMat.ToImage<Gray, Byte>();
				PredictEmguImage.ROI = PredictImageFace[0];
			Image<Gray, Byte> ROIPredictImage = PredictEmguImage.Resize(500, 500, Inter.Cubic).Convert<Gray, Byte>();
			ROIPredictImage.Save("C:\\Users\\motaz\\Desktop\\predict.jpg");
			var result = recognizer.Predict(ROIPredictImage);

			return result.Label ==0 && result.Distance < 7000 ? Ok(true) : Ok(false);
		


		}

	}
}

