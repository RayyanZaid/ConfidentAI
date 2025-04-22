// Create a component called Result that takes in a prop called result which is a map like { "key" : value, "key2" : value} and displays it

// Make it simple

import React from "react";

interface ResultProps {
  result: Record<string, any>;
}

const Result: React.FC<ResultProps> = ({ result }) => {
  return (
    <div>
      <h1>Your Interview Result</h1>
      <ul>
        {Object.entries(result).map(([key, value]) => (
          <li key={key}>
            <strong>{key}:</strong> {value}
          </li>
        ))}
      </ul>
    </div>
  );
};
export default Result;
