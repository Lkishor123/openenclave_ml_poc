// /app/page.tsx
'use client';

import { useState } from 'react';

export default function Home() {
  const [inputValue, setInputValue] = useState('');
  const [result, setResult] = useState<{ input_text: string; sentiment: string } | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');

  const [attestationLoading, setAttestationLoading] = useState(false);
  const [attestationEvidence, setAttestationEvidence] = useState('');
  const [attestationError, setAttestationError] = useState('');

  const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8080/api/analyze';

  const handleAnalyze = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    setError('');
    setResult(null);

    try {
      const response = await fetch(apiUrl, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ "input": inputValue }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Failed to get a response from the server.');
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      if (err instanceof Error) {
        setError(err.message);
      } else {
        setError('An unknown error occurred. Please try again.');
      }
    } finally {
      setIsLoading(false);
    }
  };

  const handleAttestation = async () => {
    setAttestationLoading(true);
    setAttestationError('');
    setAttestationEvidence('');

    try {
        const attestUrl = apiUrl.replace('/api/analyze', '/api/attest');

        const response = await fetch(attestUrl);

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || 'Failed to get attestation evidence.');
        }

        const data = await response.json();
        setAttestationEvidence(data.evidence_hex);

    } catch (err) {
        if (err instanceof Error) {
            setAttestationError(err.message);
        } else {
            setAttestationError('An unknown error occurred.');
        }
    } finally {
        setAttestationLoading(false);
    }
  };

  return (
    <div className="flex items-center justify-center min-h-screen bg-gray-50">
      <div className="w-full max-w-lg p-8 space-y-6 bg-white border border-gray-200 rounded-lg shadow-md">
        <h1 className="text-2xl font-bold text-gray-800">Analyze Sentiment</h1>
        <p className="text-sm text-gray-600">
            Enter a sentence to be analyzed. Your input is processed securely inside an Intel SGX enclave.
        </p>

        <form onSubmit={handleAnalyze} className="space-y-4">
          <textarea
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            placeholder="e.g., This project is incredibly helpful!"
            className="w-full px-3 py-2 text-gray-700 border rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            rows={3}
            required
          />
          <button
            type="submit"
            disabled={isLoading}
            className="w-full px-4 py-2 font-semibold text-white bg-blue-600 rounded-md hover:bg-blue-700 disabled:bg-blue-300 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
          >
            {isLoading ? 'Analyzing...' : 'Analyze Sentiment'}
          </button>
        </form>

        <div>
          <h2 className="font-semibold text-gray-700">Result:</h2>
          <div className="w-full p-3 mt-2 bg-gray-100 border rounded-md min-h-[60px] text-gray-800">
             {isLoading && <p className="text-gray-500">...</p>}
             {error && <p className="text-red-500">{error}</p>}
             {result && (
                <div>
                    <p className="break-words"><strong>Input:</strong> {result.input_text}</p>
                    <p>
                        <strong>Sentiment:</strong> <span className={`font-bold ${result.sentiment === 'Negative' ? 'text-red-600' : 'text-green-600'}`}>{result.sentiment}</span>
                    </p>
                </div>
             )}
          </div>
        </div>

        {/* --- ATTESTATION UI --- */}
        <div className="pt-4 border-t">
            <h2 className="font-semibold text-gray-700">Verify Enclave Integrity</h2>
            <p className="text-sm text-gray-600 mt-1">
                Click the button below to get cryptographic proof (attestation evidence) that the backend is running inside a genuine Intel SGX enclave.
            </p>
            <button
                onClick={handleAttestation}
                disabled={attestationLoading}
                className="w-full mt-3 px-4 py-2 font-semibold text-white bg-gray-700 rounded-md hover:bg-gray-800 disabled:bg-gray-400 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-gray-500"
            >
                {attestationLoading ? 'Generating Evidence...' : 'Get Attestation Evidence'}
            </button>
            {attestationEvidence && (
                <div className="mt-3">
                    <h3 className="font-medium text-gray-800">Attestation Quote (Hex):</h3>
                    <div className="p-2 mt-1 text-xs text-gray-600 bg-gray-100 border rounded-md break-all font-mono">
                        {attestationEvidence}
                    </div>
                    <p className="text-xs text-center mt-2 text-gray-500">
                        In a real application, this quote would be sent to a verification service like Microsoft Azure Attestation.
                    </p>
                </div>
            )}
            {attestationError && (
                <p className="mt-3 text-red-500">{attestationError}</p>
            )}
        </div>

      </div>
    </div>
  );
}
