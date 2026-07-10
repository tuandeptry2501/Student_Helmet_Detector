import React, { useState, useEffect } from 'react';
import Papa from 'papaparse';
import './App.css';

const API_URL = "http://127.0.0.1:5000/analyze";

function App() {
  const [violations, setViolations] = useState([]);
  const [search, setSearch] = useState('');
  const [selectedMedia, setSelectedMedia] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [emptyResponse, setEmptyResponse] = useState(false);

  const parseCSV = (csvText) => {
    Papa.parse(csvText, {
      header: true,
      skipEmptyLines: true,
      complete: (results) => {
        const data = results.data
          .filter((row) => row['Thời gian tải lên'] && row['Đường dẫn hình ảnh'])
          .map((row) => {
            const time = row['Thời gian tải lên'];
            const date = time.split(' ')[0];
            return {
              time,
              licensePlate: row['Biển số xe'] || 'Không xác định',
              imageUrl: row['Đường dẫn hình ảnh'],
              date,
            };
          });
        setViolations(data);
        setLoading(false);
      },
      error: (err) => {
        setError('Không đọc được CSV: ' + err.message);
        setLoading(false);
      },
    });
  };

  useEffect(() => {
    fetch('/violations.csv')
      .then((response) => {
        if (!response.ok) throw new Error('Không tải được file CSV');
        return response.text();
      })
      .then((csvText) => parseCSV(csvText))
      .catch(() => {
        console.warn('Không tìm thấy violations.csv, bỏ qua.');
        setLoading(false);
      });
  }, []);

  const handleFileUpload = (e) => {
    const file = e.target.files[0];
    if (!file) return;
    if (file.name.endsWith('.csv')) {
      const reader = new FileReader();
      reader.onload = (event) => parseCSV(event.target.result);
      reader.readAsText(file, 'UTF-8');
    } else {
      handleMediaUpload(e);
    }
  };

  const handleMediaUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    setUploading(true);
    setEmptyResponse(false);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await fetch(API_URL, {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) throw new Error('Lỗi khi gửi dữ liệu đến backend');

      const result = await res.json();
      console.log("📥 Dữ liệu backend trả về:", result);

      if (Array.isArray(result) && result.length > 0) {
        const newViolations = result.map((item) => ({
          time: item.time,
          licensePlate: item.license_plate || 'Không xác định',
          imageUrl: item.cropped_image_url,
          date: item.time.split(' ')[0],
        }));
        setViolations((prev) => [...newViolations, ...prev]);
      } else {
        console.warn("⚠️ Backend trả về mảng rỗng.");
        setEmptyResponse(true);
      }
    } catch (err) {
      setError('Phân tích thất bại: ' + err.message);
    } finally {
      setUploading(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (!file) return;
    if (file.name.endsWith('.csv')) {
      const reader = new FileReader();
      reader.onload = (event) => parseCSV(event.target.result);
      reader.readAsText(file, 'UTF-8');
    } else {
      const fakeEvent = { target: { files: [file] } };
      handleMediaUpload(fakeEvent);
    }
  };

  const handleDragOver = (e) => e.preventDefault();

  const filteredViolations = violations.filter(
    (v) =>
      v.time.toLowerCase().includes(search.toLowerCase()) ||
      v.licensePlate.toLowerCase().includes(search.toLowerCase())
  );

  const violationsByDate = filteredViolations.reduce((groups, v) => {
    if (!groups[v.date]) groups[v.date] = [];
    groups[v.date].push(v);
    return groups;
  }, {});

  if (loading) return <div>Đang tải...</div>;
  if (error) return <div className="error">{error}</div>;

  return (
    <div className="App">
      <h1>📸 DANH SÁCH VI PHẠM</h1>

      <div
        className="upload-section"
        onDrop={handleDrop}
        onDragOver={handleDragOver}
      >
        <div className="upload-box">
          <div className="upload-icon">⬆️</div>
          <p>Kéo và thả ảnh / video vào đây, hoặc:</p>
          <div className="upload-buttons">
            <label className="btn-upload">
              📂 Chọn tệp
              <input
                type="file"
                accept="image/*,video/*,.csv"
                onChange={handleFileUpload}
                hidden
              />
            </label>
          </div>
        </div>
        {uploading && <p>⏳ Đang phân tích bằng YOLO...</p>}
        {emptyResponse && <p>⚠️ Không phát hiện vi phạm nào.</p>}
      </div>

      <input
        type="text"
        placeholder="🔍 Tìm theo thời gian hoặc biển số..."
        value={search}
        onChange={(e) => setSearch(e.target.value)}
        className="search-box"
      />

      {Object.keys(violationsByDate).length === 0 ? (
        <p>📭 Chưa có dữ liệu để hiển thị.</p>
      ) : (
        Object.keys(violationsByDate)
          .sort((a, b) => b.localeCompare(a))
          .map((date) => (
            <div key={date} className="day-group">
              <h2>🗓️ {date}</h2>
              <table className="violation-table">
                <thead>
                  <tr>
                    <th>Thời gian</th>
                    <th>Ảnh</th>
                    <th>Biển số</th>
                  </tr>
                </thead>
                <tbody>
                  {violationsByDate[date].map((violation, index) => (
                    <tr key={index}>
                      <td>{violation.time}</td>
                      <td>
                        <img
                          src={violation.imageUrl}
                          alt={violation.time}
                          width="100"
                          onClick={() => setSelectedMedia(violation.imageUrl)}
                        />
                      </td>
                      <td>{violation.licensePlate}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ))
      )}

      {selectedMedia && (
        <div className="overlay" onClick={() => setSelectedMedia(null)}>
          <div className="popup">
            {selectedMedia.endsWith('.mp4') || selectedMedia.endsWith('.mov') ? (
              <video src={selectedMedia} controls autoPlay />
            ) : (
              <img src={selectedMedia} alt="Preview" />
            )}
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
