import React from 'react';
import { Link } from 'react-router-dom';

const Navbar = () => {
    return (
        <nav className="fixed top-0 left-0 right-0 bg-white/5 backdrop-blur-lg border-b border-white/10 z-50">
            <div className="max-w-6xl mx-auto px-8 py-4 flex justify-between items-center">
                <Link to="/" className="text-2xl font-bold text-blue-400">
                ⊕  Medivision 
                </Link>
                <div className="flex space-x-8">
                    <Link to="/" className="hover:text-blue-400 transition-all duration-300">
                        Home
                    </Link>
                    <Link to="/working" className="hover:text-blue-400 transition-all duration-300">
                        How It Works
                    </Link>
                    <Link to="/about" className="hover:text-blue-400 transition-all duration-300">
                        About
                    </Link>
                </div>
            </div>
        </nav>
    );
};

export default Navbar;