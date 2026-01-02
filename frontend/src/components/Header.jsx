export default function Header() {
    return (
        <header className="text-center mb-12 animate-fade-in">
            <h1 className="text-4xl md:text-5xl font-bold mb-4">
                <span className="gradient-text">Student Dropout</span>
                <br />
                <span className="text-white">Prediction System</span>
            </h1>
            <p className="text-slate-400 text-lg max-w-2xl mx-auto">
                AI-powered prediction of student outcomes using machine learning.
                Identify at-risk students early and take timely intervention.
            </p>

            {/* Feature badges */}
            <div className="flex flex-wrap justify-center gap-3 mt-6">
                <span className="px-3 py-1 bg-indigo-500/20 text-indigo-300 rounded-full text-sm border border-indigo-500/30">
                    🎯 77% Accuracy
                </span>
                <span className="px-3 py-1 bg-purple-500/20 text-purple-300 rounded-full text-sm border border-purple-500/30">
                    📊 SHAP Explanations
                </span>
                <span className="px-3 py-1 bg-pink-500/20 text-pink-300 rounded-full text-sm border border-pink-500/30">
                    ⚡ Real-time
                </span>
            </div>
        </header>
    )
}
