import { useState, useMemo } from 'react'

// Hard constraints for validation
const FIELD_CONSTRAINTS = {
    'Age': { min: 17, max: 70, type: 'number' },
    'Gender': { options: [0, 1], labels: ['Female', 'Male'], type: 'select' },
    'Marital_Status': { options: [1, 2, 3, 4, 5, 6], labels: ['Single', 'Married', 'Widower', 'Divorced', 'Facto union', 'Legally separated'], type: 'select' },
    'Course': { min: 0, max: 20, type: 'number' },
    'Mother_Qualification': { min: 0, max: 45, type: 'number' },
    'Father_Qualification': { min: 0, max: 45, type: 'number' },
    'Previous_Qualification': { min: 0, max: 20, type: 'number' },
    'Admission_Grade': { min: 0, max: 200, type: 'number' },
    'Displaced': { options: [0, 1], labels: ['No', 'Yes'], type: 'toggle' },
    'Debtor': { options: [0, 1], labels: ['No', 'Yes'], type: 'toggle' },
    'Tuition_Fees_Up_To_Date': { options: [0, 1], labels: ['No', 'Yes'], type: 'toggle' },
    'Scholarship_Holder': { options: [0, 1], labels: ['No', 'Yes'], type: 'toggle' },
    'Unemployment_Rate': { min: 0, max: 30, step: 0.1, type: 'number' },
    'Inflation_Rate': { min: -5, max: 10, step: 0.1, type: 'number' },
    'GDP': { min: -10, max: 10, step: 0.1, type: 'number' },
}

/**
 * Dynamic student information form with strict validation.
 */
export default function StudentForm({ schema, onSubmit, onReset, isLoading }) {
    const [includeExplanation, setIncludeExplanation] = useState(true)
    const [errors, setErrors] = useState({})

    // Initialize form values from schema defaults
    const initialValues = useMemo(() => {
        const values = {}
        schema.features.forEach(feature => {
            const info = schema.feature_info[feature]
            const constraint = FIELD_CONSTRAINTS[feature]
            if (constraint?.options) {
                values[feature] = constraint.options[0]
            } else {
                values[feature] = info?.default ?? 0
            }
        })
        return values
    }, [schema])

    const [formValues, setFormValues] = useState(initialValues)

    // Group features by category
    const featureGroups = useMemo(() => {
        const groups = {
            'Demographics': ['Age', 'Gender', 'Marital_Status'],
            'Academic': ['Course', 'Admission_Grade', 'Previous_Qualification',
                'Mother_Qualification', 'Father_Qualification'],
            'Financial': ['Scholarship_Holder', 'Tuition_Fees_Up_To_Date', 'Debtor', 'Displaced'],
            'Economic': ['Unemployment_Rate', 'Inflation_Rate', 'GDP'],
            'Other': []
        }

        const assigned = new Set(Object.values(groups).flat())
        schema.features.forEach(f => {
            if (!assigned.has(f)) groups['Other'].push(f)
        })

        return Object.entries(groups).filter(([_, features]) =>
            features.some(f => schema.features.includes(f))
        )
    }, [schema.features])

    const validateField = (feature, value) => {
        const constraint = FIELD_CONSTRAINTS[feature]
        if (!constraint) return null

        if (constraint.min !== undefined && value < constraint.min) {
            return `Minimum value is ${constraint.min}`
        }
        if (constraint.max !== undefined && value > constraint.max) {
            return `Maximum value is ${constraint.max}`
        }
        if (constraint.options && !constraint.options.includes(value)) {
            return `Invalid option`
        }
        return null
    }

    const handleChange = (feature, value) => {
        const numValue = parseFloat(value) || 0
        const error = validateField(feature, numValue)

        setErrors(prev => ({
            ...prev,
            [feature]: error
        }))

        setFormValues(prev => ({
            ...prev,
            [feature]: numValue
        }))
    }

    const handleSubmit = (e) => {
        e.preventDefault()

        // Validate all fields
        const newErrors = {}
        let hasErrors = false

        for (const feature of schema.features) {
            const error = validateField(feature, formValues[feature])
            if (error) {
                newErrors[feature] = error
                hasErrors = true
            }
        }

        setErrors(newErrors)

        if (hasErrors) {
            return
        }

        onSubmit(formValues, includeExplanation)
    }

    const handleReset = () => {
        setFormValues(initialValues)
        setErrors({})
        onReset?.()
    }

    const renderField = (feature) => {
        const info = schema.feature_info[feature] || {}
        const constraint = FIELD_CONSTRAINTS[feature] || {}
        const value = formValues[feature] ?? 0
        const error = errors[feature]

        // Select dropdown
        if (constraint.type === 'select') {
            return (
                <div key={feature} className="mb-4">
                    <label className="input-label">
                        {formatLabel(feature)}
                    </label>
                    <select
                        value={value}
                        onChange={(e) => handleChange(feature, parseInt(e.target.value))}
                        className={`input-field ${error ? 'ring-2 ring-red-500' : ''}`}
                    >
                        {constraint.options.map((opt, i) => (
                            <option key={opt} value={opt}>
                                {constraint.labels?.[i] || opt}
                            </option>
                        ))}
                    </select>
                    {error && <p className="text-red-400 text-xs mt-1">{error}</p>}
                </div>
            )
        }

        // Toggle switch for binary fields
        if (constraint.type === 'toggle') {
            return (
                <div key={feature} className="flex items-center justify-between py-2">
                    <label className="text-sm text-slate-300">
                        {formatLabel(feature)}
                    </label>
                    <button
                        type="button"
                        onClick={() => handleChange(feature, value === 1 ? 0 : 1)}
                        className={`relative w-14 h-7 rounded-full transition-colors duration-200 ${value === 1 ? 'bg-indigo-500' : 'bg-slate-600'
                            }`}
                    >
                        <span
                            className={`absolute top-1 left-1 w-5 h-5 bg-white rounded-full transition-transform duration-200 ${value === 1 ? 'translate-x-7' : ''
                                }`}
                        />
                    </button>
                </div>
            )
        }

        // Numeric input with validation
        return (
            <div key={feature} className="mb-4">
                <label className="input-label">
                    {formatLabel(feature)}
                    {constraint.min !== undefined && constraint.max !== undefined && (
                        <span className="text-slate-500 text-xs ml-2">
                            ({constraint.min} - {constraint.max})
                        </span>
                    )}
                </label>
                <input
                    type="number"
                    value={value}
                    onChange={(e) => handleChange(feature, e.target.value)}
                    min={constraint.min}
                    max={constraint.max}
                    step={constraint.step || 1}
                    className={`input-field ${error ? 'ring-2 ring-red-500' : ''}`}
                />
                {error && <p className="text-red-400 text-xs mt-1">{error}</p>}
            </div>
        )
    }

    const formatLabel = (feature) => {
        return feature.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())
    }

    const hasErrors = Object.values(errors).some(e => e !== null && e !== undefined)

    return (
        <form onSubmit={handleSubmit}>
            {featureGroups.map(([groupName, features]) => (
                <div key={groupName} className="mb-6">
                    <h3 className="text-sm font-semibold text-slate-400 uppercase tracking-wider mb-3 flex items-center gap-2">
                        {groupName === 'Demographics' && '👤'}
                        {groupName === 'Academic' && '🎓'}
                        {groupName === 'Financial' && '💰'}
                        {groupName === 'Economic' && '📊'}
                        {groupName === 'Other' && '📋'}
                        {groupName}
                    </h3>
                    <div className="space-y-1">
                        {features
                            .filter(f => schema.features.includes(f))
                            .map(renderField)
                        }
                    </div>
                </div>
            ))}

            {/* Explanation toggle */}
            <div className="mb-6 p-4 bg-slate-800/50 rounded-xl">
                <div className="flex items-center justify-between">
                    <div>
                        <p className="text-sm font-medium text-slate-300">Include Explanation</p>
                        <p className="text-xs text-slate-500">Show why the prediction was made (SHAP)</p>
                    </div>
                    <button
                        type="button"
                        onClick={() => setIncludeExplanation(!includeExplanation)}
                        className={`relative w-14 h-7 rounded-full transition-colors duration-200 ${includeExplanation ? 'bg-indigo-500' : 'bg-slate-600'
                            }`}
                    >
                        <span
                            className={`absolute top-1 left-1 w-5 h-5 bg-white rounded-full transition-transform duration-200 ${includeExplanation ? 'translate-x-7' : ''
                                }`}
                        />
                    </button>
                </div>
            </div>

            {/* Action buttons */}
            <div className="flex gap-3">
                <button
                    type="submit"
                    disabled={isLoading || hasErrors}
                    className="btn-primary flex-1 flex items-center justify-center gap-2"
                >
                    {isLoading ? (
                        <>
                            <div className="spinner w-5 h-5"></div>
                            Analyzing...
                        </>
                    ) : (
                        <>
                            <span>🔮</span>
                            Predict Outcome
                        </>
                    )}
                </button>
                <button
                    type="button"
                    onClick={handleReset}
                    className="btn-secondary"
                >
                    Reset
                </button>
            </div>
        </form>
    )
}
