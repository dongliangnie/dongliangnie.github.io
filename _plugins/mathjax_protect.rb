# Shields $...$ inline math from the Markdown converter.
#
# kramdown parses underscores (and other markdown characters) inside inline
# math, e.g. $H_{x}$ becomes $H<em>{x}</em>$, which breaks MathJax. Math spans
# are swapped for alphanumeric placeholders before conversion and restored in
# the rendered output.
module Jekyll
  module MathJaxProtect
    TOKEN = "MATHJAXPROTECTED%dEND".freeze
    TOKEN_PATTERN = /MATHJAXPROTECTED(\d+)END/.freeze
    CODE_TOKEN = "MATHJAXCODETOKEN%dEND".freeze
    CODE_TOKEN_PATTERN = /MATHJAXCODETOKEN(\d+)END/.freeze
    INLINE_MATH = /(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)/m.freeze

    module_function

    def protect(content)
      code = []
      math = []

      stash_code_blocks(content, code)
      protected = content.gsub(INLINE_MATH) do
        math << "$#{Regexp.last_match(1)}$"
        format(TOKEN, math.length - 1)
      end
      [restore_code_blocks(protected, code), math]
    end

    def stash_code_blocks(content, code)
      content.gsub!(/^```.*?^```[ \t]*$/m) do |match|
        code << match
        format(CODE_TOKEN, code.length - 1)
      end
      content.gsub!(/`[^`\n]+`/) do |match|
        code << match
        format(CODE_TOKEN, code.length - 1)
      end
    end

    def restore_code_blocks(content, code)
      content.gsub(CODE_TOKEN_PATTERN) { code[Regexp.last_match(1).to_i] }
    end

    def restore(content, math)
      content.gsub(TOKEN_PATTERN) { math[Regexp.last_match(1).to_i] }
    end
  end
end

Jekyll::Hooks.register [:documents, :pages], :pre_render do |doc, _payload|
  protected, math = Jekyll::MathJaxProtect.protect(doc.content.dup)
  doc.instance_variable_set(:@mathjax_math, math)
  doc.content = protected
end

Jekyll::Hooks.register [:documents, :pages], :post_render do |doc, _payload|
  math = doc.instance_variable_get(:@mathjax_math)
  next if math.nil? || math.empty?

  doc.output = Jekyll::MathJaxProtect.restore(doc.output, math)
end
