import { useState, useEffect } from 'react'
import StudentForm from './components/StudentForm'
import ResultCard from './components/ResultCard'
import ShapChart from './components/ShapChart'
import Header from './components/Header'
import { fetchSchema, predictDropout } from './api'
import './App.css'

function App() {
  const [schema, setSchema] = useState(null)
  const [loading, setLoading] = useState(true)
  const [predicting, setPredicting] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)

  // Fetch schema on mount
  useEffect(() => {
    const loadSchema = async () => {
      try {
        const data = await fetchSchema()
        setSchema(data)
        setError(null)
      } catch (err) {
        setError('Failed to connect to the prediction API. Please ensure the backend is running.')
        console.error('Schema fetch error:', err)
      } finally {
        setLoading(false)
      }
    }
    loadSchema()
  }, [])

  const handlePredict = async (formData, includeExplanation) => {
    setPredicting(true)
    setError(null)
    try {
      const prediction = await predictDropout(formData, includeExplanation)
      setResult(prediction)
    } catch (err) {
      setError(err.message || 'Prediction failed. Please try again.')
      setResult(null)
    } finally {
      setPredicting(false)
    }
  }

  const handleReset = () => {
    setResult(null)
    setError(null)
  }

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <div className="spinner mx-auto mb-4 w-12 h-12"></div>
          <p className="text-slate-400">Loading prediction system...</p>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900">
      {/* Background decoration */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="absolute -top-1/2 -left-1/4 w-96 h-96 bg-indigo-500/10 rounded-full blur-3xl"></div>
        <div className="absolute -bottom-1/2 -right-1/4 w-96 h-96 bg-purple-500/10 rounded-full blur-3xl"></div>
      </div>

      <div className="relative z-10 container mx-auto px-4 py-8 max-w-6xl">
        <Header />

        {error && (
          <div className="mb-8 p-4 bg-red-500/20 border border-red-500/30 rounded-xl text-red-400 animate-fade-in">
            <p className="flex items-center gap-2">
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              {error}
            </p>
          </div>
        )}

        <div className="grid lg:grid-cols-2 gap-8">
          {/* Form Section */}
          <div className="glass-card p-6 animate-fade-in">
            <h2 className="text-xl font-semibold mb-6 flex items-center gap-2">
              <span className="text-2xl">📝</span>
              Student Information
            </h2>

            {schema ? (
              <StudentForm
                schema={schema}
                onSubmit={handlePredict}
                onReset={handleReset}
                isLoading={predicting}
              />
            ) : (
              <p className="text-slate-400">Unable to load form schema.</p>
            )}
          </div>

          {/* Results Section */}
          <div className="space-y-6">
            {result ? (
              <>
                <ResultCard result={result} />
                {result.explanation && result.explanation.length > 0 && (
                  <ShapChart explanation={result.explanation} />
                )}
              </>
            ) : (
              <div className="glass-card p-6 animate-fade-in">
                <div className="text-center py-12">
                  <div className="text-6xl mb-4">🎓</div>
                  <h3 className="text-lg font-medium text-slate-300 mb-2">
                    Ready to Predict
                  </h3>
                  <p className="text-slate-500 text-sm">
                    Fill in the student information form and click "Predict Outcome" to see results.
                  </p>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Footer */}
        <footer className="mt-12 text-center text-slate-500 text-sm">
          <p>Student Dropout Prediction System v2.0 • Built with FastAPI + React</p>
        </footer>
      </div>
    </div>
  )
}

export default App
