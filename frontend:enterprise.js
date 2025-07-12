document.addEventListener('DOMContentLoaded', () => {
    window.addEventListener('scroll', () => {
        if (window.scrollY > document.body.scrollHeight / 3) {
            document.getElementById('demo-btn').style.display = 'block';
        }
    });

    const panels = document.querySelectorAll('.panel');
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.opacity = 1;
                entry.target.style.transform = 'translateY(0)';
            }
        });
    });
    panels.forEach(panel => observer.observe(panel));

    document.querySelector('.api-btn').addEventListener('click', () => {
        alert('API Onboarding Started');
    });
});