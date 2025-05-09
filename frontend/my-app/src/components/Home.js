import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import '../index.css';
import LungsAnim from '../assets/AnimationLungs.json';
import Lottie from 'lottie-react';

const Home = () => {
  const [file, setFile] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [imagePreview, setImagePreview] = useState(null);
  const [imageDimensions, setImageDimensions] = useState({ width: 0, height: 0 });
  const imageRef = useRef(null);

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      setFile(selectedFile);
      setImagePreview(null);
      setPrediction(null);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) return;

    setLoading(true);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post('http://127.0.0.1:8000/predict', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      setPrediction(response.data);
      setImagePreview(`data:image/png;base64,${response.data.image_base64}`);
    } catch (error) {
      console.error('Prediction error:', error);
    } finally {
      setLoading(false);
    }
  };

  // Update image dimensions when preview loads or window resizes
  useEffect(() => {
    const updateDimensions = () => {
      if (imageRef.current) {
        setImageDimensions({
          width: imageRef.current.naturalWidth,
          height: imageRef.current.naturalHeight
        });
      }
    };

    const handleResize = () => {
      updateDimensions();
    };

    if (imagePreview) {
      updateDimensions();
      window.addEventListener('resize', handleResize);
    }

    return () => {
      window.removeEventListener('resize', handleResize);
    };
  }, [imagePreview]);

  // Calculate scaled positions for infected areas
  const getScaledPosition = (area) => {
    if (!imageRef.current || !imageDimensions.width || !imageDimensions.height) {
      return area;
    }

    const scaleX = imageRef.current.clientWidth / imageDimensions.width;
    const scaleY = imageRef.current.clientHeight / imageDimensions.height;

    return {
      x: area.x * scaleX,
      y: area.y * scaleY,
      width: area.width * scaleX,
      height: area.height * scaleY
    };
  };

  return (
    <section id="home" className="px-4 sm:px-8">
      <div className="max-w-4xl mx-auto">
        <h1 className="text-3xl sm:text-4xl font-bold text-center mb-8 sm:mb-12 mt-6 sm:mt-8">
          <div className="lottie-container w-24 h-24 sm:w-32 sm:h-32 mx-auto mb-4">
            <Lottie
              animationData={LungsAnim}
              loop={true}
              className="lottie-animation"
            />
          </div>
          Lung Nodule Classification System
        </h1>

        {/* Image Preview with Responsive Highlighted Areas */}
        {imagePreview && (
          <div className="image-preview-container fixed top-4 right-4 w-[120px] sm:w-[180px] md:w-[250px] lg:w-[300px] border-2 border-white/20 rounded-lg bg-white/10 backdrop-blur-md overflow-hidden z-50 transition-all duration-300">
            <img 
              ref={imageRef}
              src={imagePreview} 
              alt="Preview" 
              className="preview-image w-full h-auto object-contain"
              onLoad={() => {
                if (imageRef.current) {
                  setImageDimensions({
                    width: imageRef.current.naturalWidth,
                    height: imageRef.current.naturalHeight
                  });
                }
              }}
            />
            {prediction?.infected_areas?.map((area, index) => {
              const scaled = getScaledPosition(area);
              return (
                <div
                  key={index}
                  className="infected-area absolute border-2 border-red-500/80 bg-red-500/20 pointer-events-none"
                  style={{
                    left: `${scaled.x}px`,
                    top: `${scaled.y}px`,
                    width: `${scaled.width}px`,
                    height: `${scaled.height}px`,
                  }}
                />
              );
            })}
          </div>
        )}
        
        <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-6 sm:p-8 shadow-xl border border-black/20">
          <form onSubmit={handleSubmit} className="space-y-6 sm:space-y-8">
            <div className="space-y-4 custom-div">
              <label className="block text-lg font-medium">
                Upload DICOM Scan
              </label>
              <div className="flex items-center justify-center w-full">
                <label className="upload-label flex flex-col items-center justify-center w-full py-8 px-4 border-2 border-dashed border-white/30 rounded-xl cursor-pointer hover:border-blue-400 transition-colors duration-300">
                  {file ? (
                    <span className="text-blue-400 font-medium text-center">
                      {file.name}
                    </span>
                  ) : (
                    <>
                      <svg
                        className="w-10 h-10 sm:w-12 sm:h-12 text-blue-400 mb-3 sm:mb-4"
                        fill="none"
                        stroke="currentColor"
                        viewBox="0 0 24 24"
                      >
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth="2"
                          d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
                        />
                      </svg>
                      <span className="text-blue-400 font-medium text-center text-sm sm:text-base">
                        Select .dcm file
                      </span>
                    </>
                  )}
                  <input
                    type="file"
                    accept=".dcm"
                    onChange={handleFileChange}
                    className="hidden"
                  />
                </label>
              </div>
            </div>

            <button
              type="submit"
              disabled={!file || loading}
              className="w-full py-3 sm:py-4 px-6 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 rounded-xl font-semibold text-lg transition-all duration-300 flex items-center justify-center"
            >
              {loading ? (
                <div className="loading-spinner border-2 border-white/30 border-t-blue-400 rounded-full w-6 h-6 animate-spin"></div>
              ) : (
                'Start Analysis'
              )}
            </button>
          </form>

          {prediction && (
            <div className="results animate-fade-in mt-6 sm:mt-8">
              <h2 className="text-2xl sm:text-3xl font-bold mb-3 sm:mb-4 flex items-center">
                {prediction.class_label === 'Malignant' ? '⚠️' : '✅'}
                <span className="ml-2 sm:ml-3">{prediction.class_label} Detection</span>
              </h2>
              <div className="space-y-2">
                <p className="text-lg sm:text-xl">
                  Confidence Level:{" "}
                  <span className="font-bold">
                    {(prediction.confidence * 100).toFixed(2)}%
                  </span>
                </p>
                <p className="text-xs sm:text-sm opacity-80">
                  {prediction.class_label === 'Malignant'
                    ? "Clinical follow-up recommended. This result requires medical validation."
                    : "Regular screening advised. This result requires medical validation."}
                </p>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Loading Animation */}
      {loading && (
        <div className="loading-container fixed inset-0 flex items-center justify-center bg-black/50 z-50">
          <div className="loading-spinner border-4 border-white/30 border-t-blue-400 rounded-full w-12 h-12 animate-spin"></div>
        </div>
      )}
    </section>
  );
};

export default Home;