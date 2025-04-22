const Spinner = () => {
  return (
    <>
      <div
        style={{
          width: "24px",
          height: "24px",
          border: "3px solid #ccc",
          borderTop: "3px solid #3498db",
          borderRadius: "50%",
          animation: "spin 1s linear infinite",
        }}
      />
      <style>
        {`
            @keyframes spin {
              0% { transform: rotate(0deg); }
              100% { transform: rotate(360deg); }
            }
          `}
      </style>
    </>
  );
};

export default Spinner;
