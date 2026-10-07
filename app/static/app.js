const form = document.getElementById('predict-form');
if (form) {
  const input = document.getElementById('text');
  const empty = document.getElementById('result-empty');
  const loading = document.getElementById('result-loading');
  const error = document.getElementById('result-error');
  const result = document.getElementById('result-data');
  const submit = form.querySelector('button[type="submit"]');
  document.querySelectorAll('.example').forEach(button => button.addEventListener('click', () => { input.value = button.dataset.text; input.focus(); }));
  form.addEventListener('submit', async event => {
    event.preventDefault();
    const text = input.value.trim();
    if (!text) { error.textContent = 'Enter a tweet or choose an example to analyze.'; error.hidden = false; return; }
    empty.hidden = true; result.hidden = true; error.hidden = true; loading.hidden = false; submit.disabled = true;
    try {
      const response = await fetch('/api/predict', {method: 'POST', headers: {'content-type': 'application/json'}, body: JSON.stringify({dataset: document.getElementById('dataset').value, text})});
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || 'Prediction failed.');
      const prediction = data.results[0], label = prediction.label_name;
      document.getElementById('verdict-label').textContent = label.charAt(0).toUpperCase() + label.slice(1);
      document.getElementById('verdict-icon').className = `verdict-icon ${label}`;
      document.getElementById('confidence').textContent = `${(prediction.confidence * 100).toFixed(1)}%`;
      document.getElementById('model-used').textContent = `${data.model.replaceAll('_', ' ')} · ${data.features.toUpperCase()}`;
      document.getElementById('latency').textContent = `${data.latency_ms} ms`;
      const bars = document.getElementById('probabilities'); bars.replaceChildren();
      for (const [name, probability] of Object.entries(prediction.probabilities)) {
        const row = document.createElement('div'); row.className = 'probability-row';
        const labels = document.createElement('div'); labels.className = 'probability-labels';
        const title = document.createElement('span'); title.textContent = name.charAt(0).toUpperCase() + name.slice(1);
        const value = document.createElement('strong'); value.textContent = `${(probability * 100).toFixed(1)}%`; labels.append(title, value);
        const track = document.createElement('div'); track.className = 'bar-track';
        const fill = document.createElement('div'); fill.className = `bar-fill ${name}`; fill.style.width = `${Math.max(0, Math.min(100, probability * 100))}%`;
        track.append(fill); row.append(labels, track); bars.append(row);
      }
      result.hidden = false;
    } catch (err) { error.textContent = err.message || 'Unable to analyze this post.'; error.hidden = false; }
    finally { loading.hidden = true; submit.disabled = false; }
  });
}
