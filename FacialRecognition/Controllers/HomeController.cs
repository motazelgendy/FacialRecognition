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
		public async Task<IActionResult> MatchFaces(List <IFormFile> Listface, IFormFile PredictSample)
		{

			string HaarCascadePath = Directory.GetCurrentDirectory() + @"\haarcascade_frontalface_alt.xml";
			CascadeClassifier faceCasacdeClassifier = new CascadeClassifier(HaarCascadePath);
			EigenFaceRecognizer recognizer = new EigenFaceRecognizer();
			VectorOfMat TrainImageVecofMat = new VectorOfMat();
			List<int> Labels = new List<int>();
			VectorOfInt vectorOfInt = new VectorOfInt();

			
			List<Image<Gray, Byte>> EmguImages = new List<Image<Gray, byte>>();
			for (int i=0; i< Listface.Count;i++)
			{
			Labels.Add(i);
			using var stream = Listface[i].OpenReadStream();
		    using var MemoryStream = new MemoryStream();
			stream.CopyTo(MemoryStream);
			byte[] TrainBytes = MemoryStream.ToArray();
			Mat TrainMat = new Mat();
			CvInvoke.Imdecode(TrainBytes, ImreadModes.Color, TrainMat);
			Rectangle[] CorppedFace = faceCasacdeClassifier.DetectMultiScale(TrainMat, 1.1, 3, Size.Empty, Size.Empty);
			Image<Gray, Byte> TrainEmguImage = TrainMat.ToImage<Gray, Byte>();
			TrainEmguImage.ROI = CorppedFace[0];
			Image<Gray, Byte> ROIImage = TrainEmguImage.Resize(500, 500, Inter.Cubic).Convert<Gray, Byte>();
			EmguImages.Add(ROIImage);	
			}


			TrainImageVecofMat.Push(EmguImages.ToArray());
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
			var result = recognizer.Predict(ROIPredictImage);

			return result.Label ==0 && result.Distance < 7000 ? Ok(true) : Ok(false);
		


		}

	}
}

