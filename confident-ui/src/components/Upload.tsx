import {
  Trash as TrashIcon,
  Upload as UploadIcon,
} from "@phosphor-icons/react";
import { CSSProperties, useRef, useState } from "react";

import InterviewImage from "../assets/InterviewImage.png";
import Result from "./Result";
import Spinner from "./Spinner";
const styles: Record<string, CSSProperties> = {
  submitButton: {
    padding: "16px 32px", // Bigger padding
    fontSize: "18px", // Larger font
    fontWeight: "bold",
    borderRadius: "8px",
    border: "none",
    backgroundColor: "#007bff",
    color: "#fff",
    cursor: "pointer",
  },
};

function UploadVideo() {
  const inputRef = useRef<HTMLInputElement>(null);

  const [file, setFile] = useState<File | null>(null);
  const [status, setStatus] = useState<
    "pending" | "uploading" | "success" | "error"
  >("pending");

  const [videoURL, setVideoURL] = useState<string | null>(null);

  const handleUpload = () => {
    setStatus("uploading");
    setTimeout(() => {
      setStatus("success");
    }, 2000);
  };
  return (
    <form
      onSubmit={(e) => {
        e.preventDefault();
        handleUpload();
      }}
      onReset={() => {
        setStatus("pending");
        setFile(null);
      }}
    >
      {status === "uploading" ? (
        <div
          style={{ display: "flex", justifyContent: "center", padding: "2rem" }}
        >
          <Spinner />
        </div>
      ) : (
        <>
          {!file && (
            <div>
              <h1 style={styles.heading}>Upload Your Interview Video Here</h1>
              <img
                src={InterviewImage}
                alt="Interview"
                style={{ maxWidth: "100%", height: "auto" }}
              />
            </div>
          )}

          <div
            style={styles.uploadBox}
            onClick={() => inputRef.current?.click()}
          >
            <div style={styles.content}>
              {status === "success" && (
                <Result
                  result={{
                    key: "value",
                    key2: "value",
                  }}
                />
              )}
              {status === "error" && (
                <p style={styles.errorText}>File upload failed!</p>
              )}

              <div style={{ height: "50px" }} />

              {videoURL && (
                <video
                  src={videoURL}
                  controls
                  style={{
                    width: "100%",
                    maxHeight: "400px",
                    marginTop: "1rem",
                  }}
                />
              )}
              <div style={{ height: "50px" }} />
              <div style={styles.circle}>
                <UploadIcon size={32} />
              </div>

              <h2 style={styles.uploadText}>
                Drop your files here,
                <br /> or browse
              </h2>
            </div>
          </div>

          <input
            type="file"
            onChange={(e) => {
              setStatus("pending");
              const selectedFile = e.target.files?.[0] || null;
              setFile(selectedFile);
              if (selectedFile) {
                setVideoURL(URL.createObjectURL(selectedFile));
              } else {
                setVideoURL(null);
              }
            }}
            hidden
            ref={inputRef}
          />

          {file && (
            <>
              <div style={styles.fileList}>
                <div style={styles.fileElement}>
                  <div style={styles.filler} />

                  {status === "pending" && (
                    <button
                      style={styles.trash}
                      onClick={(e) => {
                        e.preventDefault();
                        if (inputRef.current) {
                          inputRef.current.value = "";
                        }
                        setFile(null);
                      }}
                    >
                      <TrashIcon size={32} />
                    </button>
                  )}
                </div>
              </div>

              <div style={styles.uploadButtonContainer}>
                {status === "success" ? (
                  <button style={styles.submitButton} type="reset">
                    Reset
                  </button>
                ) : (
                  <div>
                    <div style={{ height: "50px" }} />
                    <button style={styles.submitButton} type="submit">
                      Submit
                    </button>
                  </div>
                )}
              </div>
            </>
          )}
        </>
      )}
    </form>
  );
}

export default UploadVideo;
