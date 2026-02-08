# final_fix.py
print("=== FIXING UNICODE CHARACTERS ===")

with open('demo_module1_fixed.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace fancy Unicode characters with simple text
replacements = {
    "✓": "[OK]",
    "📊": "[DATA]",
    "🤖": "[MODEL]", 
    "📈": "[VOLUME]",
    "💧": "[LIQUIDITY]",
    "📡": "[API]",
    "📤": "[RESPONSE]",
    "📄": "[FILE]",
    "📓": "[EXAMPLE]",
    "🔬": "[TEST]",
    "🏆": "[WINNER]"
}

for old, new in replacements.items():
    if old in content:
        content = content.replace(old, new)
        print(f"Replaced: {old} -> {new}")

# Also fix the encoding for file writing
content = content.replace("with open('./module1_summary.txt', 'w') as f:", 
                         "with open('./module1_summary.txt', 'w', encoding='utf-8') as f:")
content = content.replace("with open('./example_usage.py', 'w') as f:", 
                         "with open('./example_usage.py', 'w', encoding='utf-8') as f:")

with open('demo_module1_fixed.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("\n✅ Unicode characters fixed!")
print("✅ File encoding issues resolved!")
print("\nNow run: python demo_module1_fixed.py")