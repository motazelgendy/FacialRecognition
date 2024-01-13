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


namespace FacialRecognition.Controllers
{
	[Route("api/[controller]")]
	[ApiController]
	public class HomeController : ControllerBase
	{

		[HttpPut("MatchFaces")]
		public async Task<IActionResult> MatchFaces(IFormFile FirstSample, IFormFile SecondSample, IFormFile PredictSample)
		{
			EigenFaceRecognizer recognizer = new EigenFaceRecognizer();
			VectorOfMat TrainImageVecofMat = new VectorOfMat();
			List<int> Labels = new List<int>();
			
			VectorOfInt vectorOfInt = new VectorOfInt();
			Labels.Add(1);
			Labels.Add(2);

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
			Image<Gray, Byte> FitstTrainEmguImage = FirstTrainMat.ToImage<Gray, Byte>().Resize(500, 500, Inter.Cubic);
			Image<Gray, Byte> SecondTrainEmguImage = SecondTrainMat.ToImage<Gray, Byte>().Resize(500, 500, Inter.Cubic);
			List<Image<Gray, Byte>> TrainedFaces = new List<Image<Gray, Byte>>();
			TrainedFaces.Add(FitstTrainEmguImage);
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

			Image<Gray, byte> PredictEmguImage = PredictMat.ToImage<Bgr, byte>().Resize(500, 500, Inter.Cubic).Convert<Gray, byte>();
			var result = recognizer.Predict(PredictEmguImage);

			return result.Distance > 3000 ? Ok(true) : Ok(false);
		


		}

	}
}
