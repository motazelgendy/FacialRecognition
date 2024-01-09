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

namespace FacialRecognition.Controllers
{
	[Route("api/[controller]")]
	[ApiController]
	public class HomeController : ControllerBase
	{

		[HttpPut("MatchFaces")]
		public async Task<IActionResult> MatchFaces(IFormFile TrainImage, IFormFile RandomImage, IFormFile PredictImage)
		{
			EigenFaceRecognizer recognizer = new EigenFaceRecognizer();
			VectorOfMat TrainImageVecofMat = new VectorOfMat();
			List<int> Labels = new List<int>();
			
			VectorOfInt vectorOfInt = new VectorOfInt();
			Labels.Add(1);
			Labels.Add(2);


			using var StreamTrainImage = TrainImage.OpenReadStream();
			using var StreamRandomImage = RandomImage.OpenReadStream();
			var TrainbmpImage = new Bitmap(StreamTrainImage);
			var RandombmpImage = new Bitmap(StreamRandomImage);
			Image<Gray, Byte> TrainEmguImage = TrainbmpImage.ToImage<Gray, Byte>().Resize(500, 500, Inter.Cubic);
			Image<Gray, Byte> RandomEmguImage = RandombmpImage.ToImage<Gray, Byte>().Resize(500, 500, Inter.Cubic);

			List<Image<Gray, Byte>> TrainedFaces = new List<Image<Gray, Byte>>();
			TrainedFaces.Add(TrainEmguImage);
			TrainedFaces.Add(RandomEmguImage);



			TrainImageVecofMat.Push(TrainedFaces.ToArray());
			vectorOfInt.Push(Labels.ToArray());

			recognizer.Train(TrainImageVecofMat, vectorOfInt);

			using var PredictTrainImage = PredictImage.OpenReadStream();
			var PredictbmpImage = new Bitmap(PredictTrainImage);
			Image<Gray, byte> PredictEmguImage = PredictbmpImage.ToImage<Bgr, byte>().Resize(500, 500, Inter.Cubic).Convert<Gray, byte>();
			var result = recognizer.Predict(PredictEmguImage);

			return result.Distance > 3000 ? Ok(true) : Ok(false);
		


		}

	}
}
