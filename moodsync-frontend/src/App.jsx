import React, { useState, useEffect } from 'react';
import axios from 'axios';

function App() {
    const [selectedFile, setSelectedFile] = useState(null);
    const [uploadResult, setUploadResult] = useState(null);
    const [loading, setLoading] = useState(false);

    const handleFileChange = (e) => {
        setSelectedFile(e.target.files[0]);
    };

    const handleUpload = async () => {
        try {
            setLoading(true);

            const formData = new FormData();
            formData.append('file', selectedFile);

            const response = await axios.post('http://localhost:5000/upload', formData, {
                headers: {
                    'Content-Type': 'multipart/form-data',
                },
            });

            const responseData = JSON.parse(response.data.lambda_result.body);
            setUploadResult(responseData.data);
            console.log(responseData.data);
        } catch (error) {
            console.error('Error uploading file:', error);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="bg-gray-100 min-h-screen">
            {/* Header */}
            <header className="bg-blue-500 text-white py-4">
                <div className="container mx-auto text-center">
                    <h1 className="text-4xl font-bold mb-2">MoodSync</h1>
                    <p className="text-sm">Your mood, your playlist.</p>
                </div>
            </header>

            {/* Main Content */}
            <main className="container mx-auto mt-8">
                <h2 className="text-2xl font-bold mb-4">Capture Your Vibe, Discover Your Playlist</h2>
                <div className="flex items-center space-x-4 mt-4">
                    <input type="file" onChange={handleFileChange} className="py-2 px-4 border rounded-lg" />
                    <button onClick={handleUpload} className="bg-green-500 hover:bg-green-600 text-white py-2 px-6 rounded-lg">
                        Analyze Mood
                    </button>
                </div>
                <h2 className="text-2xl text-gray-700 font-bold mb-4  mt-8">Featured music tailored for you</h2>

                {/* Display featured music list */}
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-8 mt-8">
                    {uploadResult?.map((music) => (
                        <div key={music.id} className="bg-white p-4 rounded-lg shadow-md">
                            <img src={music.album_cover_url} alt="Song Cover" className="w-full h-48 object-cover mb-4 rounded-md" />
                            <p className="text-lg font-bold">{music.track_name}</p>
                            <p className="text-gray-600">{music.artist}</p>
                            <p className="text-blue-500 underline">
                                <a href={music.track_url} target="_blank" rel="noopener noreferrer">
                                    Listen on Spotify
                                </a>
                            </p>
                        </div>
                    ))}
                </div>
            </main>

            {/* Footer */}
            <footer className="bg-blue-500 text-white py-4 mt-8">
                <div className="container mx-auto text-center">
                    <p>&copy; 2024 MoodSync. All rights reserved.</p>
                </div>
            </footer>
        </div>
    );
}

export default App;
