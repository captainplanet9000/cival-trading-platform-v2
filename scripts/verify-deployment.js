#!/usr/bin/env node

/**
 * Deployment Verification Script
 * Comprehensive checks for Railway deployment readiness
 */

const fs = require('fs')
const path = require('path')
const { execSync } = require('child_process')

console.log('🚀 Cival Trading Platform - Deployment Verification')
console.log('=' * 50)

const checks = []

// Check 1: Verify package.json exists and has required scripts
function checkPackageJson() {
  try {
    const packagePath = path.join(process.cwd(), 'package.json')
    const packageJson = JSON.parse(fs.readFileSync(packagePath, 'utf8'))
    
    const requiredScripts = ['build', 'start', 'dev', 'lint']
    const missingScripts = requiredScripts.filter(script => !packageJson.scripts[script])
    
    if (missingScripts.length === 0) {
      checks.push({ name: 'Package.json Scripts', status: '✅', message: 'All required scripts present' })
    } else {
      checks.push({ name: 'Package.json Scripts', status: '❌', message: `Missing scripts: ${missingScripts.join(', ')}` })
    }
    
    // Check dependencies
    const criticalDeps = ['next', 'react', 'react-dom', 'typescript']
    const missingDeps = criticalDeps.filter(dep => !packageJson.dependencies[dep])
    
    if (missingDeps.length === 0) {
      checks.push({ name: 'Critical Dependencies', status: '✅', message: 'All critical dependencies present' })
    } else {
      checks.push({ name: 'Critical Dependencies', status: '❌', message: `Missing: ${missingDeps.join(', ')}` })
    }
    
  } catch (error) {
    checks.push({ name: 'Package.json', status: '❌', message: 'Package.json not found or invalid' })
  }
}

// Check 2: Verify Next.js configuration
function checkNextConfig() {
  try {
    const nextConfigPath = path.join(process.cwd(), 'next.config.js')
    if (fs.existsSync(nextConfigPath)) {
      const config = fs.readFileSync(nextConfigPath, 'utf8')
      
      if (config.includes('standalone')) {
        checks.push({ name: 'Next.js Config', status: '✅', message: 'Standalone output configured for Railway' })
      } else {
        checks.push({ name: 'Next.js Config', status: '⚠️', message: 'Consider adding standalone output for optimal deployment' })
      }
    } else {
      checks.push({ name: 'Next.js Config', status: '⚠️', message: 'next.config.js not found - using defaults' })
    }
  } catch (error) {
    checks.push({ name: 'Next.js Config', status: '❌', message: 'Error reading next.config.js' })
  }
}

// Check 3: Verify Railway configuration
function checkRailwayConfig() {
  try {
    const railwayConfigPath = path.join(process.cwd(), 'railway.toml')
    if (fs.existsSync(railwayConfigPath)) {
      const config = fs.readFileSync(railwayConfigPath, 'utf8')
      
      if (config.includes('buildCommand') && config.includes('startCommand')) {
        checks.push({ name: 'Railway Config', status: '✅', message: 'Railway.toml properly configured' })
      } else {
        checks.push({ name: 'Railway Config', status: '⚠️', message: 'Railway.toml missing required commands' })
      }
    } else {
      checks.push({ name: 'Railway Config', status: '⚠️', message: 'railway.toml not found - Railway will use defaults' })
    }
  } catch (error) {
    checks.push({ name: 'Railway Config', status: '❌', message: 'Error reading railway.toml' })
  }
}

// Check 4: Verify environment template
function checkEnvironmentTemplate() {
  try {
    const envTemplatePath = path.join(process.cwd(), 'env.template')
    if (fs.existsSync(envTemplatePath)) {
      const template = fs.readFileSync(envTemplatePath, 'utf8')
      
      const requiredVars = ['NEXT_PUBLIC_API_URL', 'DATABASE_URL', 'REDIS_URL']
      const hasRequired = requiredVars.every(varName => template.includes(varName))
      
      if (hasRequired) {
        checks.push({ name: 'Environment Template', status: '✅', message: 'Environment template with required variables' })
      } else {
        checks.push({ name: 'Environment Template', status: '⚠️', message: 'Environment template missing some variables' })
      }
    } else {
      checks.push({ name: 'Environment Template', status: '⚠️', message: 'env.template not found' })
    }
  } catch (error) {
    checks.push({ name: 'Environment Template', status: '❌', message: 'Error reading env.template' })
  }
}

// Check 5: Verify TypeScript compilation
function checkTypeScript() {
  try {
    console.log('Checking TypeScript compilation...')
    execSync('npx tsc --noEmit --skipLibCheck', { stdio: 'pipe' })
    checks.push({ name: 'TypeScript', status: '✅', message: 'TypeScript compilation successful' })
  } catch (error) {
    checks.push({ name: 'TypeScript', status: '❌', message: 'TypeScript compilation errors found' })
  }
}

// Check 6: Verify build process
function checkBuild() {
  try {
    console.log('Testing build process...')
    execSync('npm run build', { stdio: 'pipe' })
    checks.push({ name: 'Build Process', status: '✅', message: 'Build completed successfully' })
  } catch (error) {
    checks.push({ name: 'Build Process', status: '❌', message: 'Build failed - check for errors' })
  }
}

// Check 7: Verify Docker configuration
function checkDocker() {
  try {
    const dockerfilePath = path.join(process.cwd(), 'Dockerfile')
    if (fs.existsSync(dockerfilePath)) {
      const dockerfile = fs.readFileSync(dockerfilePath, 'utf8')
      
      if (dockerfile.includes('FROM node:') && dockerfile.includes('EXPOSE')) {
        checks.push({ name: 'Docker Config', status: '✅', message: 'Dockerfile properly configured' })
      } else {
        checks.push({ name: 'Docker Config', status: '⚠️', message: 'Dockerfile may need improvements' })
      }
    } else {
      checks.push({ name: 'Docker Config', status: 'ℹ️', message: 'Dockerfile not found - not required for Railway' })
    }
  } catch (error) {
    checks.push({ name: 'Docker Config', status: '❌', message: 'Error reading Dockerfile' })
  }
}

// Check 8: Verify monorepo structure
function checkMonorepoStructure() {
  try {
    const expectedDirs = ['src', 'public', 'docs']
    const missingDirs = expectedDirs.filter(dir => !fs.existsSync(path.join(process.cwd(), dir)))
    
    if (missingDirs.length === 0) {
      checks.push({ name: 'Monorepo Structure', status: '✅', message: 'All expected directories present' })
    } else {
      checks.push({ name: 'Monorepo Structure', status: '⚠️', message: `Missing directories: ${missingDirs.join(', ')}` })
    }
    
    // Check for Python backend
    const pythonBackendPath = path.join(process.cwd(), 'python-ai-services')
    if (fs.existsSync(pythonBackendPath)) {
      checks.push({ name: 'Python Backend', status: '✅', message: 'Python AI services directory found' })
    } else {
      checks.push({ name: 'Python Backend', status: 'ℹ️', message: 'Python backend not found - frontend only deployment' })
    }
  } catch (error) {
    checks.push({ name: 'Monorepo Structure', status: '❌', message: 'Error checking directory structure' })
  }
}

// Run all checks
async function runAllChecks() {
  console.log('Running deployment verification checks...\n')
  
  checkPackageJson()
  checkNextConfig()
  checkRailwayConfig()
  checkEnvironmentTemplate()
  checkMonorepoStructure()
  checkDocker()
  
  // These checks take longer, run them last
  if (process.argv.includes('--full')) {
    checkTypeScript()
    checkBuild()
  }
  
  // Display results
  console.log('\n📋 Verification Results:')
  console.log('=' * 50)
  
  checks.forEach(check => {
    console.log(`${check.status} ${check.name}: ${check.message}`)
  })
  
  // Summary
  const passed = checks.filter(c => c.status === '✅').length
  const warnings = checks.filter(c => c.status === '⚠️').length
  const failed = checks.filter(c => c.status === '❌').length
  const info = checks.filter(c => c.status === 'ℹ️').length
  
  console.log('\n📊 Summary:')
  console.log(`✅ Passed: ${passed}`)
  console.log(`⚠️ Warnings: ${warnings}`)
  console.log(`❌ Failed: ${failed}`)
  console.log(`ℹ️ Info: ${info}`)
  
  if (failed === 0) {
    console.log('\n🎉 Deployment verification completed! Ready for Railway deployment.')
    console.log('\nNext steps:')
    console.log('1. Set up environment variables in Railway dashboard')
    console.log('2. Connect your GitHub repository to Railway')
    console.log('3. Deploy with: railway up')
  } else {
    console.log('\n⚠️ Please fix the failed checks before deploying.')
    process.exit(1)
  }
}

// Run the verification
runAllChecks().catch(error => {
  console.error('Error running verification:', error)
  process.exit(1)
})