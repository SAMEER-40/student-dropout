const API_BASE = '/api'

/**
 * Fetch feature schema from backend.
 * Frontend renders forms dynamically from this - never hardcode features.
 */
export async function fetchSchema() {
    const response = await fetch(`${API_BASE}/schema`)
    if (!response.ok) {
        throw new Error(`Failed to fetch schema: ${response.statusText}`)
    }
    return response.json()
}

/**
 * Make a dropout prediction.
 * @param {Object} features - Dictionary of feature name to value
 * @param {boolean} explain - Include SHAP explanation
 * @param {number} topK - Number of features in explanation
 */
export async function predictDropout(features, explain = true, topK = 10) {
    const response = await fetch(`${API_BASE}/predict`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            features,
            explain,
            top_k_features: topK
        })
    })

    if (!response.ok) {
        const error = await response.json().catch(() => ({ detail: response.statusText }))
        throw new Error(error.detail || 'Prediction failed')
    }

    return response.json()
}

/**
 * Check API health status.
 */
export async function checkHealth() {
    const response = await fetch(`${API_BASE}/health`)
    return response.json()
}
