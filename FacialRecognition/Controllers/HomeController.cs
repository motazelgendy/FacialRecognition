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
		public async Task<IActionResult> MatchFaces(IFormFile TrainImage, IFormFile PredictImage)
		{
			EigenFaceRecognizer recognizer = new EigenFaceRecognizer();
			VectorOfMat TrainImageVecofMat = new VectorOfMat();
			List<int> Labels = new List<int>();
			Labels.Add(1);
			VectorOfInt vectorOfInt = new VectorOfInt();


			using var StreamTrainImage = TrainImage.OpenReadStream();
			var TrainbmpImage = new Bitmap(StreamTrainImage);
			Image<Gray, Byte> TrainEmguImage = TrainbmpImage.ToImage<Bgr, Byte>().Resize(200, 200, Inter.Cubic).Convert<Gray, Byte>();
			List<Image<Gray, Byte>> TrainedFaces = new List<Image<Gray, byte>>();
			TrainedFaces.Add(TrainEmguImage);
			TrainImageVecofMat.Push(TrainedFaces.ToArray());
			vectorOfInt.Push(Labels.ToArray());

			recognizer.Train(TrainImageVecofMat, vectorOfInt);

			using var PredictTrainImage = PredictImage.OpenReadStream();
			var PredictbmpImage = new Bitmap(PredictTrainImage);
			Image<Gray, Byte> PredictEmguImage = PredictbmpImage.ToImage<Bgr, Byte>().Resize(200, 200, Inter.Cubic).Convert<Gray, Byte>();
			var result = recognizer.Predict(PredictEmguImage);
			if (result.Distance > 3000)
			{
				return Ok(true);
			}

			else
			{
				return Ok(false);
			}
		


		}

	}
}
