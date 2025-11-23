"""
Скрипт для проверки и исправления конфигурации Ollama в .env
"""
import re
from pathlib import Path

env_file = Path(".env")

if not env_file.exists():
    print("❌ Файл .env не найден")
    exit(1)

content = env_file.read_text(encoding='utf-8')
lines = content.split('\n')

print("=" * 70)
print("Проверка конфигурации Ollama в .env")
print("=" * 70)

issues_found = False
fixed_lines = []

for i, line in enumerate(lines, 1):
    # Проверяем строки с OLLAMA_MODEL
    if 'OLLAMA_MODEL' in line and '=' in line:
        # Ищем модели с форматом квантования
        if re.search(r':\d+b-[a-z0-9_]+', line):
            issues_found = True
            print(f"\n⚠️  Строка {i}: Найдена модель с форматом квантования")
            print(f"   Было: {line.strip()}")
            
            # Убираем формат квантования
            fixed_line = re.sub(r'(-q\d+[a-z0-9_-]+)', '', line)
            fixed_line = fixed_line.strip()
            
            print(f"   Стало: {fixed_line}")
            fixed_lines.append((i, line, fixed_line))
            lines[i-1] = fixed_line
        else:
            # Проверяем правильность имени модели
            model_match = re.search(r'OLLAMA_MODEL[^=]*=([^\s#]+)', line)
            if model_match:
                model_name = model_match.group(1).strip()
                print(f"✓ Строка {i}: {model_name}")

if issues_found:
    print("\n" + "=" * 70)
    print("Найдены проблемы! Исправить автоматически? (y/n): ", end='')
    response = input().strip().lower()
    
    if response == 'y':
        # Создаем backup
        backup_file = Path(".env.backup")
        backup_file.write_text(content, encoding='utf-8')
        print(f"✓ Создан backup: {backup_file}")
        
        # Сохраняем исправленный файл
        env_file.write_text('\n'.join(lines), encoding='utf-8')
        print("✓ Файл .env обновлен")
    else:
        print("❌ Изменения не применены")
else:
    print("\n✅ Все модели указаны правильно!")

print("=" * 70)


