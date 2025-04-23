// Create a component called Result that takes in a prop called result which is a map like { "key" : value, "key2" : value} and displays it

// Make it simple

import React from "react";

interface ResultProps {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  result: string;
}

const Result: React.FC<ResultProps> = ({ result }) => {
  return (
    <div>
      <h1>Your Interview Result</h1>
      <h2>{result}</h2>
    </div>
  );
};
export default Result;
