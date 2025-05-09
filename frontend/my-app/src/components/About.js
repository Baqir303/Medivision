import React from 'react';

const About = () => {
    return (
        <section id="about" className="py-20 px-8 mt-20">
            <div className="max-w-4xl mx-auto">
                <h2 className="text-3xl font-bold mb-8 text-center">About This Project</h2>
                <div className="bg-white/10 p-6 rounded-xl">
                    <p className="opacity-80 mb-4">
                        This project is designed to classify lung nodules using advanced AI techniques. It leverages a Capsule Network (CapsNet) model to analyze DICOM images and provide predictions with confidence levels.
                    </p>
                    <p className="opacity-80 mb-4">
                        The system is intended for research purposes only and should not be used for clinical diagnosis without further validation.
                    </p>
                    <p className="opacity-80">
                        Built with ❤️ using React, FastAPI, and PyTorch.
                    </p>
                </div>
            </div>
        </section>
    );
};

export default About;