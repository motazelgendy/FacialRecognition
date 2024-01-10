using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;
using Emgu.CV;
using Emgu.CV.Structure;
using Emgu.CV.Face;
using System.Threading.Tasks;
using Emgu.CV.Util;
using Emgu.CV.Reg;
using System.Drawing;
using Emgu.CV.CvEnum;
using System;
using static System.Net.Mime.MediaTypeNames;
using System.Collections.Generic;
using Microsoft.AspNetCore.Identity;

namespace FacialRecognition.Controllers
{
	[Route("api/[controller]")]
	[ApiController]
	public class HomeController : ControllerBase
	{

		[HttpPut("MatchFaces")]
		public async Task<IActionResult> MatchFaces(IFormFile RandomImage, IFormFile TrainImage, IFormFile PredictImage)
		{
			EigenFaceRecognizer recognizer = new EigenFaceRecognizer();
			VectorOfMat TrainImageVecofMat = new VectorOfMat();
			List<int> Labels = new List<int>();
			
			VectorOfInt vectorOfInt = new VectorOfInt();
			Labels.Add(1);
			Labels.Add(2);

			using var RandomStream = RandomImage.OpenReadStream();
			using var ImageStream = TrainImage.OpenReadStream();
			using var MemoryrandomStream = new MemoryStream();
			using var MemoryTrainStream = new MemoryStream();
			RandomStream.CopyTo(MemoryrandomStream);
			ImageStream.CopyTo(MemoryTrainStream);
			byte[] RandomBytes = MemoryrandomStream.ToArray();
			byte[] TrainBytes = MemoryTrainStream.ToArray();

			Mat RandomMat = new Mat();
			Mat TrainMat =new Mat();
			CvInvoke.Imdecode(RandomBytes, ImreadModes.Color, RandomMat);
			CvInvoke.Imdecode(TrainBytes, ImreadModes.Color, TrainMat);
			Image<Gray, Byte> RandomEmguImage = RandomMat.ToImage<Gray, Byte>().Resize(500, 500, Inter.Cubic);
			Image<Gray, Byte> TrainEmguImage = TrainMat.ToImage<Gray, Byte>().Resize(500, 500, Inter.Cubic);
			List<Image<Gray, Byte>> TrainedFaces = new List<Image<Gray, Byte>>();
			TrainedFaces.Add(TrainEmguImage);
			TrainedFaces.Add(RandomEmguImage);
		
			



			TrainImageVecofMat.Push(TrainedFaces.ToArray());
			vectorOfInt.Push(Labels.ToArray());

			recognizer.Train(TrainImageVecofMat, vectorOfInt);

			using var PredictStream = PredictImage.OpenReadStream();
			using var MemoryPredictStream = new MemoryStream();
			PredictStream.CopyTo(MemoryPredictStream);
			byte[] PredictBytes = MemoryrandomStream.ToArray();
			Mat PredictMat = new Mat();
			CvInvoke.Imdecode(PredictBytes, ImreadModes.Color, PredictMat);

			Image<Gray, byte> PredictEmguImage = PredictMat.ToImage<Bgr, byte>().Resize(500, 500, Inter.Cubic).Convert<Gray, byte>();
			var result = recognizer.Predict(PredictEmguImage);

			return result.Distance > 3000 && result.Label == 1  ? Ok(true) : Ok(false);
		


		}

	}
}
